"""Bundle builder.

This module implements a pragmatic "single file" bundler:

- At build time, it downloads wheels for the target platform and extracts them
  into a ``deps/`` directory.
- It copies the app sources into an ``app/`` directory.
- It zips both directories, base64-encodes the zip bytes, and writes a single
  Python file containing a small bootstrapper that extracts and runs the app.
"""

from dataclasses import dataclass
import base64
import hashlib
import io
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import time
import zipfile

from python_flattener.target import TargetConfig


class BuildError(RuntimeError):
    """Raised when bundling fails."""


@dataclass(frozen=True, slots=True)
class EntryPoint:
    """Represents how the bundled app should be executed.

    :ivar kind: Either ``path`` or ``module``.
    :ivar path: Path relative to the staged ``app/`` directory (kind=path).
    :ivar module: Module name to run (kind=module).
    """

    kind: str
    path: str | None
    module: str | None


@dataclass(frozen=True, slots=True)
class AppLayout:
    """Prepared app staging layout.

    :ivar app_dir: Staged app root directory (added to ``sys.path`` at runtime).
    :ivar entry: Resolved entrypoint.
    """

    app_dir: pathlib.Path
    entry: EntryPoint


@dataclass(frozen=True, slots=True)
class CopyStats:
    """Stats collected while copying a directory tree.

    :ivar files_copied: Number of files copied.
    :ivar bytes_copied: Total bytes copied (best-effort).
    """

    files_copied: int
    bytes_copied: int


def _validate_compresslevel(compresslevel: int) -> None:
    """Validate a zip compression level.

    :param compresslevel: Compression level (0-9).
    :raises BuildError: If the level is out of range.
    """

    if compresslevel < 0 or compresslevel > 9:
        raise BuildError(f"Invalid compresslevel={compresslevel}; expected 0-9.")


def _resolve_cache_root(cache_dir: pathlib.Path | None) -> pathlib.Path:
    """Resolve the build cache directory.

    Defaults to ``.python_flattener_cache`` under the current working directory.

    :param cache_dir: Optional cache directory override.
    :returns: Cache root directory.
    """

    if cache_dir is not None:
        root: pathlib.Path = cache_dir
    else:
        root = pathlib.Path.cwd() / ".python_flattener_cache"

    root.mkdir(parents=True, exist_ok=True)
    return root


def _compute_stage_excludes(
    *,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    cache_dir: pathlib.Path | None,
) -> set[str]:
    """Compute relative paths to exclude while staging app sources.

    :param input_path: Input file or directory.
    :param output_path: Output bundle path.
    :param cache_dir: Optional cache directory.
    :returns: Set of relative paths (POSIX-style) to exclude.
    """

    root: pathlib.Path
    if input_path.is_dir() is True:
        root = input_path
    else:
        root = input_path.parent

    root_resolved: pathlib.Path = root.resolve()
    excludes: set[str] = set()

    out_resolved: pathlib.Path = output_path.resolve()
    if out_resolved.is_relative_to(root_resolved) is True:
        excludes.add(out_resolved.relative_to(root_resolved).as_posix())

    cache_resolved: pathlib.Path = _resolve_cache_root(cache_dir).resolve()
    if cache_resolved.is_relative_to(root_resolved) is True:
        excludes.add(cache_resolved.relative_to(root_resolved).as_posix())

    return excludes


def _sha256_file(path: pathlib.Path) -> str:
    """Hash a file with SHA-256.

    :param path: File to hash.
    :returns: Hex digest.
    """

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk: bytes = f.read(1024 * 1024)
            if len(chunk) == 0:
                break
            h.update(chunk)
    return h.hexdigest()


def _target_key(target: TargetConfig) -> str:
    """Build a stable cache key for a target config.

    :param target: Target configuration.
    :returns: A filesystem-friendly key string.
    """

    return f"{target.platform_tag}__{target.python_version}__{target.implementation}__{target.abi}"


def _ensure_wheelhouse(
    *,
    requirements_path: pathlib.Path,
    cache_root: pathlib.Path,
    target: TargetConfig,
    logger: logging.Logger,
) -> list[pathlib.Path]:
    """Ensure wheels are available in the local build cache.

    :param requirements_path: requirements.txt file.
    :param cache_root: Cache root.
    :param target: Target config.
    :param logger: Logger for progress output.
    :returns: Wheel file paths.
    :raises BuildError: If dependency download/build fails.
    """

    req_hash: str = _sha256_file(requirements_path)
    wheelhouse_dir: pathlib.Path = cache_root / "wheelhouse" / _target_key(target) / req_hash
    marker: pathlib.Path = wheelhouse_dir / ".ok"

    if marker.is_file() is True:
        cached: list[pathlib.Path] = sorted(wheelhouse_dir.glob("*.whl"))
        logger.info(f"python-flattener: wheelhouse cache hit ({len(cached)} wheels)")
        if logger.isEnabledFor(logging.DEBUG) is True:
            logger.debug(f"python-flattener: wheelhouse_dir={wheelhouse_dir}")
        return cached

    logger.info("python-flattener: wheelhouse cache miss; downloading wheels with pip")
    if logger.isEnabledFor(logging.DEBUG) is True:
        logger.debug(f"python-flattener: wheelhouse_dir={wheelhouse_dir}")

    wheelhouse_dir.mkdir(parents=True, exist_ok=True)
    t0: float = time.perf_counter()
    wheel_files: list[pathlib.Path] = _download_wheels(
        requirements_path=requirements_path,
        wheel_dir=wheelhouse_dir,
        target=target,
        logger=logger,
    )
    t1: float = time.perf_counter()
    marker.write_text("ok\n", encoding="utf-8")
    logger.info(f"python-flattener: wheel download complete ({len(wheel_files)} wheels) in {t1 - t0:.2f}s")
    return wheel_files


def _zip_dir_to_path(*, root: pathlib.Path, out_path: pathlib.Path, compresslevel: int) -> None:
    """Zip a directory tree to a zip file on disk.

    :param root: Root directory to archive.
    :param out_path: Output zip path.
    :param compresslevel: Deflate compression level.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        out_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
    ) as zf:
        paths: list[pathlib.Path] = []
        for p in root.rglob("*"):
            if p.is_file() is True:
                paths.append(p)
        for p in sorted(paths):
            arcname: str = str(p.relative_to(root)).replace(os.sep, "/")
            zf.write(p, arcname=arcname)


def _ensure_deps_layer_zip(
    *,
    wheel_files: list[pathlib.Path],
    cache_root: pathlib.Path,
    requirements_path: pathlib.Path,
    target: TargetConfig,
    compresslevel: int,
    logger: logging.Logger,
) -> pathlib.Path | None:
    """Build (and cache) a deps layer zip from a set of wheels.

    This layer is reused across builds to avoid re-extracting and re-zipping
    dependency trees.

    :param wheel_files: Wheel file paths.
    :param cache_root: Cache root.
    :param requirements_path: requirements.txt file (used for cache keying).
    :param target: Target config.
    :param compresslevel: Deflate compression level.
    :param logger: Logger for progress output.
    :returns: Path to cached deps layer zip, or ``None`` if there are no wheels.
    """

    if len(wheel_files) == 0:
        logger.info("python-flattener: deps layer skipped (no wheels)")
        return None

    req_hash: str = _sha256_file(requirements_path)
    layer_dir: pathlib.Path = cache_root / "deps_layer" / _target_key(target) / req_hash
    zip_path: pathlib.Path = layer_dir / "deps_layer.zip"
    marker: pathlib.Path = layer_dir / ".ok"

    if marker.is_file() is True and zip_path.is_file() is True:
        zip_size: int = zip_path.stat().st_size
        logger.info(
            f"python-flattener: deps layer cache hit ({zip_size / (1024 * 1024):.1f} MiB)"
        )
        if logger.isEnabledFor(logging.DEBUG) is True:
            logger.debug(f"python-flattener: deps_layer_zip={zip_path}")
        return zip_path

    logger.info(f"python-flattener: deps layer cache miss; building from {len(wheel_files)} wheels")
    if logger.isEnabledFor(logging.DEBUG) is True:
        logger.debug(f"python-flattener: deps_layer_dir={layer_dir}")

    layer_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip: pathlib.Path = layer_dir / "deps_layer.zip.tmp"
    t0: float = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="deps_layer_build_", dir=layer_dir) as td:
        build_root: pathlib.Path = pathlib.Path(td)
        deps_dir: pathlib.Path = build_root / "deps"
        deps_dir.mkdir(parents=True, exist_ok=True)
        _install_wheels(wheel_files=wheel_files, deps_dir=deps_dir, target=target)
        _zip_dir_to_path(root=deps_dir, out_path=tmp_zip, compresslevel=compresslevel)
    t1: float = time.perf_counter()

    tmp_zip.replace(zip_path)
    marker.write_text("ok\n", encoding="utf-8")
    zip_size2: int = zip_path.stat().st_size
    logger.info(
        f"python-flattener: deps layer built ({zip_size2 / (1024 * 1024):.1f} MiB) in {t1 - t0:.2f}s"
    )
    return zip_path


def _build_payload_zip_bytes(
    *,
    app_dir: pathlib.Path,
    deps_layer_zip: pathlib.Path | None,
    compresslevel: int,
) -> bytes:
    """Build the outer payload zip (app + deps layer) as bytes.

    :param app_dir: Staged ``app/`` directory.
    :param deps_layer_zip: Optional cached deps layer zip.
    :param compresslevel: Deflate compression level.
    :returns: Zip bytes.
    """

    buf: io.BytesIO = io.BytesIO()
    with zipfile.ZipFile(
        buf,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
    ) as zf:
        paths: list[pathlib.Path] = []
        for p in app_dir.rglob("*"):
            if p.is_file() is True:
                paths.append(p)
        for p in sorted(paths):
            rel: str = str(p.relative_to(app_dir)).replace(os.sep, "/")
            zf.write(p, arcname=f"app/{rel}")

        if deps_layer_zip is not None:
            zf.write(
                deps_layer_zip,
                arcname="deps_layer.zip",
                compress_type=zipfile.ZIP_STORED,
            )

    return buf.getvalue()


def build_single_file(
    *,
    input_path: pathlib.Path,
    requirements_path: pathlib.Path,
    output_path: pathlib.Path,
    target: TargetConfig,
    entry_relpath: str | None,
    entry_module: str | None,
    logger: logging.Logger | None = None,
    cache_dir: pathlib.Path | None = None,
    compresslevel: int = 1,
) -> None:
    """Build a single-file bundle.

    :param input_path: Python file or directory containing the app.
    :param requirements_path: requirements.txt path (can be empty).
    :param output_path: Output path for the bundled .py file.
    :param target: Target interpreter/platform config for wheel resolution.
    :param entry_relpath: Optional entry file path relative to the input directory.
    :param entry_module: Optional module name to run (python -m style).
    :param logger: Optional logger for realtime build progress output.
    :raises BuildError: If bundling fails.
    """

    if logger is None:
        logger = logging.getLogger("python_flattener")

    if input_path.exists() is False:
        raise BuildError(f"Input path does not exist: {input_path}")
    if requirements_path.exists() is False:
        raise BuildError(f"requirements.txt does not exist: {requirements_path}")
    if requirements_path.is_file() is False:
        raise BuildError(f"requirements path is not a file: {requirements_path}")

    if entry_relpath is not None and entry_module is not None:
        raise BuildError("Use at most one of --entry and --module.")

    _validate_compresslevel(compresslevel)

    t_total0: float = time.perf_counter()
    logger.info(f"python-flattener: input={input_path}")
    logger.info(f"python-flattener: requirements={requirements_path}")
    logger.info(f"python-flattener: output={output_path}")
    logger.info(
        "python-flattener: target="
        f"{target.platform_tag} py={target.python_version} impl={target.implementation} abi={target.abi}"
    )

    with tempfile.TemporaryDirectory(prefix="python_flattener_build_") as td:
        build_root: pathlib.Path = pathlib.Path(td)
        staging_root: pathlib.Path = build_root / "staging"
        app_dir: pathlib.Path = staging_root / "app"
        staging_root.mkdir(parents=True, exist_ok=True)
        app_dir.mkdir(parents=True, exist_ok=True)

        exclude_relpaths: set[str] = _compute_stage_excludes(
            input_path=input_path,
            output_path=output_path,
            cache_dir=cache_dir,
        )
        if logger.isEnabledFor(logging.DEBUG) is True:
            logger.debug(f"python-flattener: staging excludes={sorted(exclude_relpaths)}")

        t_stage0: float = time.perf_counter()
        app_layout: AppLayout = _stage_app(
            input_path=input_path,
            app_dir=app_dir,
            entry_relpath=entry_relpath,
            entry_module=entry_module,
            exclude_relpaths=exclude_relpaths,
            logger=logger,
        )
        t_stage1: float = time.perf_counter()
        logger.info(f"python-flattener: staged app in {t_stage1 - t_stage0:.2f}s")

        cache_root: pathlib.Path = _resolve_cache_root(cache_dir)
        logger.info(f"python-flattener: cache_root={cache_root}")
        wheel_files: list[pathlib.Path] = _ensure_wheelhouse(
            requirements_path=requirements_path,
            cache_root=cache_root,
            target=target,
            logger=logger,
        )
        deps_layer_zip: pathlib.Path | None = _ensure_deps_layer_zip(
            wheel_files=wheel_files,
            cache_root=cache_root,
            requirements_path=requirements_path,
            target=target,
            compresslevel=compresslevel,
            logger=logger,
        )

        t_payload0: float = time.perf_counter()
        bundle_zip_bytes: bytes = _build_payload_zip_bytes(
            app_dir=app_dir,
            deps_layer_zip=deps_layer_zip,
            compresslevel=compresslevel,
        )
        t_payload1: float = time.perf_counter()
        logger.info(
            f"python-flattener: payload zip built ({len(bundle_zip_bytes) / (1024 * 1024):.1f} MiB) "
            f"in {t_payload1 - t_payload0:.2f}s"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        t_write0: float = time.perf_counter()
        _write_single_file(
            output_path=output_path,
            bundle_zip_bytes=bundle_zip_bytes,
            entry=app_layout.entry,
            target=target,
            logger=logger,
        )
        t_write1: float = time.perf_counter()
        out_size: int = output_path.stat().st_size
        logger.info(
            f"python-flattener: wrote {output_path} ({out_size / (1024 * 1024):.1f} MiB) in {t_write1 - t_write0:.2f}s"
        )

    t_total1: float = time.perf_counter()
    logger.info(f"python-flattener: done in {t_total1 - t_total0:.2f}s")


def _stage_app(
    *,
    input_path: pathlib.Path,
    app_dir: pathlib.Path,
    entry_relpath: str | None,
    entry_module: str | None,
    exclude_relpaths: set[str],
    logger: logging.Logger,
) -> AppLayout:
    """Copy application sources into the staging directory and resolve entrypoint.

    :param input_path: File or directory input.
    :param app_dir: Destination staging ``app/`` directory.
    :param entry_relpath: Optional entry file path relative to input directory.
    :param entry_module: Optional module name to run.
    :param logger: Logger for progress output.
    :returns: App layout information.
    :raises BuildError: If the app cannot be staged.
    """

    if input_path.is_file() is True:
        if entry_relpath is not None or entry_module is not None:
            raise BuildError("For file inputs, do not use --entry/--module; the file is the entry.")

        src_root: pathlib.Path = input_path.parent
        logger.info(f"python-flattener: staging file input; copying from {src_root}")
        stats: CopyStats = _copy_tree_app(src=src_root, dst=app_dir, exclude_relpaths=exclude_relpaths)
        logger.info(
            f"python-flattener: staged sources ({stats.files_copied} files, {stats.bytes_copied / (1024 * 1024):.1f} MiB)"
        )

        rel: str = input_path.name
        return AppLayout(
            app_dir=app_dir,
            entry=EntryPoint(kind="path", path=rel, module=None),
        )

    if input_path.is_dir() is False:
        raise BuildError(f"Unsupported input path type (expected file/dir): {input_path}")

    is_package_dir: bool = (input_path / "__init__.py").is_file()
    staged_root: pathlib.Path
    staged_entry_prefix: str
    if is_package_dir is True:
        staged_root = app_dir / input_path.name
        staged_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"python-flattener: staging package dir input; copying from {input_path}")
        stats_pkg: CopyStats = _copy_tree_app(
            src=input_path,
            dst=staged_root,
            exclude_relpaths=exclude_relpaths,
        )
        logger.info(
            f"python-flattener: staged sources ({stats_pkg.files_copied} files, {stats_pkg.bytes_copied / (1024 * 1024):.1f} MiB)"
        )
        staged_entry_prefix = input_path.name
    else:
        staged_root = app_dir
        logger.info(f"python-flattener: staging dir input; copying from {input_path}")
        stats_dir: CopyStats = _copy_tree_app(
            src=input_path,
            dst=staged_root,
            exclude_relpaths=exclude_relpaths,
        )
        logger.info(
            f"python-flattener: staged sources ({stats_dir.files_copied} files, {stats_dir.bytes_copied / (1024 * 1024):.1f} MiB)"
        )
        staged_entry_prefix = ""

    if entry_module is not None:
        return AppLayout(
            app_dir=app_dir,
            entry=EntryPoint(kind="module", path=None, module=entry_module),
        )

    if entry_relpath is not None:
        entry_path: pathlib.Path
        if len(staged_entry_prefix) > 0:
            entry_path = app_dir / staged_entry_prefix / entry_relpath
            entry_rel: str = f"{staged_entry_prefix}/{entry_relpath}"
        else:
            entry_path = app_dir / entry_relpath
            entry_rel = entry_relpath

        if entry_path.is_file() is False:
            raise BuildError(f"--entry did not exist after staging: {entry_rel!r}")
        return AppLayout(
            app_dir=app_dir,
            entry=EntryPoint(kind="path", path=entry_rel, module=None),
        )

    # Auto-detect defaults.
    if is_package_dir is True:
        pkg_main: pathlib.Path = input_path / "__main__.py"
        if pkg_main.is_file() is True:
            return AppLayout(
                app_dir=app_dir,
                entry=EntryPoint(kind="module", path=None, module=input_path.name),
            )

        pkg_main2: pathlib.Path = input_path / "main.py"
        if pkg_main2.is_file() is True:
            return AppLayout(
                app_dir=app_dir,
                entry=EntryPoint(kind="path", path=f"{input_path.name}/main.py", module=None),
            )

        raise BuildError(
            "Directory input looks like a package (has __init__.py) but has no __main__.py or main.py. "
            "Provide --module or --entry."
        )

    root_main: pathlib.Path = input_path / "__main__.py"
    if root_main.is_file() is True:
        return AppLayout(
            app_dir=app_dir,
            entry=EntryPoint(kind="path", path="__main__.py", module=None),
        )

    root_main2: pathlib.Path = input_path / "main.py"
    if root_main2.is_file() is True:
        return AppLayout(
            app_dir=app_dir,
            entry=EntryPoint(kind="path", path="main.py", module=None),
        )

    raise BuildError(
        "Directory input has no auto-detected entry. Add __main__.py or main.py, or pass --entry/--module."
    )


def _copy_tree_app(*, src: pathlib.Path, dst: pathlib.Path, exclude_relpaths: set[str]) -> CopyStats:
    """Copy an app directory tree to a destination, applying a conservative ignore list.

    :param src: Source directory.
    :param dst: Destination directory.
    :param exclude_relpaths: Relative paths within ``src`` to exclude.
    :returns: Copy statistics.
    """

    ignore_names: set[str] = {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        ".python_flattener_cache",
    }
    exclude_parts: list[tuple[str, ...]] = []
    for relpath in sorted(exclude_relpaths):
        exclude_parts.append(pathlib.PurePosixPath(relpath).parts)

    def is_excluded(relpath: pathlib.PurePosixPath) -> bool:
        """Check if a relative path should be excluded.

        :param relpath: Path relative to ``src`` (POSIX).
        :returns: ``True`` if it matches an excluded prefix.
        """

        rel_tuple: tuple[str, ...] = relpath.parts
        for ex in exclude_parts:
            if len(rel_tuple) >= len(ex) and rel_tuple[0 : len(ex)] == ex:
                return True
        return False

    def looks_like_generated_bundle(path: pathlib.Path) -> bool:
        """Check if a file appears to be a python-flattener generated bundle.

        This prevents accidentally re-bundling prior outputs when staging an app
        from a directory containing previously generated bundles.

        :param path: Candidate file path.
        :returns: ``True`` if it looks like a generated bundle.
        """

        if path.suffix != ".py":
            return False

        try:
            size: int = path.stat().st_size
        except OSError:
            return False

        if size < 64 * 1024:
            return False

        try:
            with open(path, "rb") as f:
                head: bytes = f.read(4096)
        except OSError:
            return False

        return b"This file was generated by python-flattener" in head

    files_copied: int = 0
    bytes_copied: int = 0

    dst.mkdir(parents=True, exist_ok=True)

    for root_str, dirs, files in os.walk(src, topdown=True):
        root_path: pathlib.Path = pathlib.Path(root_str)
        rel_root: pathlib.Path = root_path.relative_to(src)
        rel_root_posix: pathlib.PurePosixPath = pathlib.PurePosixPath(rel_root.as_posix())
        at_src_root: bool = rel_root_posix.as_posix() == "."

        if is_excluded(rel_root_posix) is True:
            dirs[:] = []
            continue

        keep_dirs: list[str] = []
        for d in dirs:
            if at_src_root is True and d in ignore_names:
                continue
            if is_excluded(rel_root_posix / d) is True:
                continue
            keep_dirs.append(d)
        dirs[:] = keep_dirs

        out_dir: pathlib.Path = dst / rel_root
        out_dir.mkdir(parents=True, exist_ok=True)

        for name in files:
            if name == ".DS_Store":
                continue
            if at_src_root is True and name in ignore_names:
                continue

            src_path: pathlib.Path = root_path / name
            if src_path.suffix in {".pyc", ".pyo"}:
                continue
            if looks_like_generated_bundle(src_path) is True:
                continue
            if is_excluded(rel_root_posix / name) is True:
                continue

            dest_path: pathlib.Path = dst / rel_root / name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            files_copied += 1
            try:
                bytes_copied += src_path.stat().st_size
            except OSError:
                pass

    return CopyStats(files_copied=files_copied, bytes_copied=bytes_copied)


def _copy_tree_all(*, src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy a directory tree without filtering.

    :param src: Source directory.
    :param dst: Destination directory.
    """

    for p in src.rglob("*"):
        rel: pathlib.Path = p.relative_to(src)
        if p.is_dir() is True:
            (dst / rel).mkdir(parents=True, exist_ok=True)
            continue
        if p.is_file() is True:
            target_path: pathlib.Path = dst / rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target_path)


def _download_wheels(
    *,
    requirements_path: pathlib.Path,
    wheel_dir: pathlib.Path,
    target: TargetConfig,
    logger: logging.Logger,
) -> list[pathlib.Path]:
    """Download wheels (and possibly sdists) for the given requirements.

    :param requirements_path: requirements.txt file.
    :param wheel_dir: Directory to download artifacts into.
    :param target: Target config for pip environment emulation.
    :param logger: Logger for progress output.
    :returns: List of wheel files to install.
    :raises BuildError: If dependencies cannot be resolved into usable wheels.
    """

    host_platform: str = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    host_pyver: str = f"{sys.version_info.major}.{sys.version_info.minor}"
    host_impl: str
    if sys.implementation.name == "cpython":
        host_impl = "cp"
    elif sys.implementation.name == "pypy":
        host_impl = "pp"
    else:
        host_impl = sys.implementation.name[0:2]
    host_abi: str = f"cp{sys.version_info.major}{sys.version_info.minor:02d}"

    constraints_needed: bool = (
        target.platform_tag != host_platform
        or target.python_version != host_pyver
        or target.implementation != host_impl
        or target.abi != host_abi
    )

    if constraints_needed is True:
        _pip(
            [
                "download",
                "--disable-pip-version-check",
                "--only-binary",
                ":all:",
                "--dest",
                str(wheel_dir),
                "--requirement",
                str(requirements_path),
                "--platform",
                target.platform_tag,
                "--python-version",
                target.python_version,
                "--implementation",
                target.implementation,
                "--abi",
                target.abi,
            ],
            logger=logger,
        )
    else:
        _pip(
            [
                "download",
                "--disable-pip-version-check",
                "--dest",
                str(wheel_dir),
                "--requirement",
                str(requirements_path),
            ],
            logger=logger,
        )

    wheel_files: list[pathlib.Path] = sorted(wheel_dir.glob("*.whl"))
    other_files: list[pathlib.Path] = []
    for p in sorted(wheel_dir.iterdir()):
        if p.is_file() is False:
            continue
        if p.suffix == ".whl":
            continue
        if p.name.endswith(".metadata") is True:
            continue
        if p.suffix == ".json":
            continue
        other_files.append(p)

    if len(other_files) == 0:
        return wheel_files

    built_wheels: list[pathlib.Path] = []
    for sdist in other_files:
        before: set[str] = {p.name for p in wheel_dir.glob("*.whl")}
        _pip(
            [
                "wheel",
                "--disable-pip-version-check",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
                str(sdist),
            ],
            logger=logger,
        )
        after: set[str] = {p.name for p in wheel_dir.glob("*.whl")}
        new_names: set[str] = after - before
        for name in sorted(new_names):
            built_wheels.append(wheel_dir / name)

    if constraints_needed is True:
        raise BuildError(
            "Internal error: constrained download produced non-wheel artifacts. "
            "This should not happen when using --only-binary=:all:."
        )

    if host_platform != target.platform_tag:
        non_universal: list[str] = []
        for whl in built_wheels:
            if _wheel_is_universal(whl.name) is False:
                non_universal.append(whl.name)
        if len(non_universal) > 0:
            joined: str = "\n".join(f"- {n}" for n in non_universal)
            raise BuildError(
                "Cross-platform build downloaded source distributions that did not build universal wheels.\n"
                "These would not be reliable for the target platform:\n"
                f"{joined}\n"
                "Fix: ensure wheels exist for the target, or avoid sdists for platform-specific packages."
            )

    wheel_files2: list[pathlib.Path] = sorted(wheel_dir.glob("*.whl"))
    return wheel_files2


def _wheel_is_universal(wheel_filename: str) -> bool:
    """Check if a wheel filename looks universal (``*-none-any.whl``).

    :param wheel_filename: Wheel file name (not a full path).
    :returns: ``True`` if it appears to be universal.
    """

    if wheel_filename.endswith(".whl") is False:
        return False
    parts: list[str] = wheel_filename.split("-")
    if len(parts) < 5:
        return False
    tag_part: str = parts[-1]
    # tag_part is like "py3-none-any.whl"
    return tag_part.startswith("py") is True and tag_part.endswith("none-any.whl") is True


def _install_wheels(*, wheel_files: list[pathlib.Path], deps_dir: pathlib.Path, target: TargetConfig) -> None:
    """Install wheels into the staging deps directory.

    :param wheel_files: Wheel file paths.
    :param deps_dir: Destination directory.
    :param target: Target config (currently informational).
    :raises BuildError: If wheel installation fails.
    """

    if len(wheel_files) == 0:
        return

    deps_dir.mkdir(parents=True, exist_ok=True)
    for whl in wheel_files:
        _install_wheel_file(wheel_path=whl, deps_dir=deps_dir)


def _install_wheel_file(*, wheel_path: pathlib.Path, deps_dir: pathlib.Path) -> None:
    """Install a wheel by extracting it into ``deps_dir``.

    This is not a full wheel "installer", but it handles the common cases well:
    root packages + ``.dist-info`` and the ``.data/purelib|platlib`` relocation.

    :param wheel_path: Path to ``.whl`` file.
    :param deps_dir: Destination directory.
    :raises BuildError: If extraction fails.
    """

    if wheel_path.suffix != ".whl":
        raise BuildError(f"Not a wheel: {wheel_path}")

    with tempfile.TemporaryDirectory(prefix="python_flattener_wheel_") as td:
        tmp_root: pathlib.Path = pathlib.Path(td)
        try:
            with zipfile.ZipFile(wheel_path, "r") as zf:
                zf.extractall(tmp_root)
        except zipfile.BadZipFile as e:
            raise BuildError(f"Bad wheel zip: {wheel_path}") from e

        data_dirs: list[pathlib.Path] = []
        for child in tmp_root.iterdir():
            if child.name.endswith(".data") is True and child.is_dir() is True:
                data_dirs.append(child)
                continue
            _copy_item(src=child, dst=deps_dir / child.name)

        for data_dir in data_dirs:
            purelib: pathlib.Path = data_dir / "purelib"
            platlib: pathlib.Path = data_dir / "platlib"
            if purelib.exists() is True:
                _copy_tree_all(src=purelib, dst=deps_dir)
            if platlib.exists() is True:
                _copy_tree_all(src=platlib, dst=deps_dir)

            other_root: pathlib.Path = deps_dir / "__wheel_data__" / data_dir.name
            for sub in ("data", "scripts", "headers"):
                subdir: pathlib.Path = data_dir / sub
                if subdir.exists() is True:
                    _copy_tree_all(src=subdir, dst=other_root / sub)


def _copy_item(*, src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy a file or directory to a destination.

    :param src: Source path.
    :param dst: Destination path.
    """

    if src.is_dir() is True:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    if src.is_file() is True:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return


def _pip(args: list[str], *, logger: logging.Logger | None) -> None:
    """Invoke pip with the current interpreter.

    :param args: Arguments after ``-m pip``.
    :param logger: Optional logger for debug output.
    :raises BuildError: If pip fails.
    """

    cmd: list[str] = [sys.executable, "-m", "pip", *args]
    if logger is not None and logger.isEnabledFor(logging.DEBUG) is True:
        logger.debug(f"python-flattener: running pip: {' '.join(cmd)}")

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise BuildError(f"pip invocation failed (exit={proc.returncode}): {' '.join(cmd)}")


def _zip_dir(root: pathlib.Path) -> bytes:
    """Zip a directory into bytes.

    :param root: Root directory to archive.
    :returns: Zip archive bytes.
    """

    buf: io.BytesIO = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        paths: list[pathlib.Path] = []
        for p in root.rglob("*"):
            if p.is_file() is True:
                paths.append(p)
        for p in sorted(paths):
            arcname: str = str(p.relative_to(root)).replace(os.sep, "/")
            zf.write(p, arcname=arcname)
    return buf.getvalue()


def _write_single_file(
    *,
    output_path: pathlib.Path,
    bundle_zip_bytes: bytes,
    entry: EntryPoint,
    target: TargetConfig,
    logger: logging.Logger,
) -> None:
    """Write the final self-extracting bundle script to disk.

    This avoids constructing a massive in-memory string for the base64 payload.

    :param output_path: Output path for the bundled .py file.
    :param bundle_zip_bytes: Zipped payload bytes (outer payload zip).
    :param entry: Entrypoint information.
    :param target: Target config embedded for runtime validation.
    :param logger: Logger for progress output.
    :raises BuildError: If the runtime template placeholders are missing.
    """

    sha256: str = hashlib.sha256(bundle_zip_bytes).hexdigest()
    b64_bytes: bytes = base64.b64encode(bundle_zip_bytes)

    target_json: str = (
        "{\n"
        f'  "platform_tag": "{target.platform_tag}",\n'
        f'  "python_version": "{target.python_version}",\n'
        f'  "implementation": "{target.implementation}",\n'
        f'  "abi": "{target.abi}"\n'
        "}"
    )

    entry_json: str
    if entry.kind == "module":
        if entry.module is None:
            raise BuildError("Internal error: module entry missing module name.")
        entry_json = "{\n" f'  "kind": "module",\n' f'  "module": "{entry.module}"\n' "}"
    else:
        if entry.path is None:
            raise BuildError("Internal error: path entry missing path.")
        entry_json = "{\n" f'  "kind": "path",\n' f'  "path": "{entry.path}"\n' "}"

    runtime: str = _RUNTIME_TEMPLATE
    runtime = runtime.replace("__PYFLAT_SHA256__", sha256)
    runtime = runtime.replace("__PYFLAT_TARGET_JSON__", target_json)
    runtime = runtime.replace("__PYFLAT_ENTRY_JSON__", entry_json)

    marker: str = "__PYFLAT_BUNDLE_B64__"
    idx: int = runtime.find(marker)
    if idx < 0:
        raise BuildError("Internal error: runtime template missing __PYFLAT_BUNDLE_B64__ marker.")

    prefix: str = runtime[0:idx]
    suffix: str = runtime[idx + len(marker) :]

    if logger.isEnabledFor(logging.DEBUG) is True:
        logger.debug(
            f"python-flattener: rendering output (payload_b64={len(b64_bytes) / (1024 * 1024):.1f} MiB)"
        )

    wrap_width: int = 88
    with open(output_path, "wb") as f:
        f.write(prefix.encode("utf-8"))
        i: int = 0
        n: int = len(b64_bytes)
        while i < n:
            j: int = i + wrap_width
            if j >= n:
                f.write(b64_bytes[i:n])
                break
            f.write(b64_bytes[i:j])
            f.write(b"\n")
            i = j
        f.write(suffix.encode("utf-8"))


def _render_single_file(*, bundle_zip_bytes: bytes, entry: EntryPoint, target: TargetConfig) -> str:
    """Render the final self-extracting bundle script.

    :param bundle_zip_bytes: Zipped staging directory.
    :param entry: Entrypoint information.
    :param target: Target config embedded for runtime validation.
    :returns: Python source code for the bundle.
    """

    sha256: str = hashlib.sha256(bundle_zip_bytes).hexdigest()
    b64: str = base64.b64encode(bundle_zip_bytes).decode("ascii")

    target_json: str = (
        "{\n"
        f'  "platform_tag": "{target.platform_tag}",\n'
        f'  "python_version": "{target.python_version}",\n'
        f'  "implementation": "{target.implementation}",\n'
        f'  "abi": "{target.abi}"\n'
        "}"
    )

    entry_json: str
    if entry.kind == "module":
        if entry.module is None:
            raise BuildError("Internal error: module entry missing module name.")
        entry_json = "{\n" f'  "kind": "module",\n' f'  "module": "{entry.module}"\n' "}"
    else:
        if entry.path is None:
            raise BuildError("Internal error: path entry missing path.")
        entry_json = "{\n" f'  "kind": "path",\n' f'  "path": "{entry.path}"\n' "}"

    # Keep line lengths reasonable for editors and git diffs.
    b64_wrapped: str = "\n".join(textwrap.wrap(b64, width=88))

    runtime: str = _RUNTIME_TEMPLATE
    runtime = runtime.replace("__PYFLAT_SHA256__", sha256)
    runtime = runtime.replace("__PYFLAT_BUNDLE_B64__", b64_wrapped)
    runtime = runtime.replace("__PYFLAT_TARGET_JSON__", target_json)
    runtime = runtime.replace("__PYFLAT_ENTRY_JSON__", entry_json)
    return runtime


_RUNTIME_TEMPLATE: str = textwrap.dedent(
    r'''
    #!/usr/bin/env python3
    # This file was generated by python-flattener. It is intentionally "one huge file".
    #
    # It contains a base64-encoded zip payload (app sources + deps). At runtime it
    # extracts to a cache dir and executes the entrypoint.
    #
    # Target config is embedded below in _TARGET / _ENTRY.

    import base64
    import hashlib
    import importlib.util
    import io
    import os
    import pathlib
    import platform
    import runpy
    import shutil
    import site
    import sys
    import tempfile
    import zipfile


    _BUNDLE_SHA256: str = "__PYFLAT_SHA256__"
    _TARGET: dict[str, str] = __PYFLAT_TARGET_JSON__
    _ENTRY: dict[str, str] = __PYFLAT_ENTRY_JSON__

    _BUNDLE_B64: str = r"""__PYFLAT_BUNDLE_B64__"""


    def _runtime_error(message: str) -> None:
        """Exit with a message.

        :param message: Error message.
        """

        sys.stderr.write(message)
        if message.endswith("\n") is False:
            sys.stderr.write("\n")
        raise SystemExit(2)


    def _normalize_arch(machine: str) -> str:
        """Normalize a machine string into a small set of expected values.

        :param machine: Raw machine string (e.g. from ``platform.machine()``).
        :returns: Normalized architecture string.
        """

        m: str = machine.lower()
        if m == "amd64":
            return "x86_64"
        if m == "x86_64":
            return "x86_64"
        if m == "aarch64":
            return "aarch64"
        if m == "arm64":
            return "arm64"
        if m == "armv7l":
            return "armv7l"
        if m == "i386" or m == "i686":
            return "i686"
        return m


    def _check_runtime_compat() -> None:
        """Validate interpreter compatibility with the bundle.

        :raises SystemExit: If the runtime does not match the bundle target.
        """

        expected_py: str = _TARGET["python_version"]
        actual_py: str = f"{sys.version_info.major}.{sys.version_info.minor}"
        if actual_py != expected_py:
            _runtime_error(
                "Python version mismatch for this bundle.\n"
                f"Expected: {expected_py}\n"
                f"Actual:   {actual_py}\n"
            )

        expected_impl: str = _TARGET["implementation"]
        actual_impl: str
        if sys.implementation.name == "cpython":
            actual_impl = "cp"
        elif sys.implementation.name == "pypy":
            actual_impl = "pp"
        else:
            actual_impl = sys.implementation.name[0:2]
        if actual_impl != expected_impl:
            _runtime_error(
                "Python implementation mismatch for this bundle.\n"
                f"Expected: {expected_impl}\n"
                f"Actual:   {actual_impl}\n"
            )

        expected_platform: str = _TARGET["platform_tag"]
        expected_os: str | None = None
        expected_arch: str | None = None

        parts: list[str] = expected_platform.split("_")

        if expected_platform.startswith("macosx_") is True:
            expected_os = "darwin"
            if len(parts) >= 4:
                expected_arch = "_".join(parts[3:])
        elif expected_platform.startswith("win_") is True:
            expected_os = "win32"
            if len(parts) >= 2:
                expected_arch = "_".join(parts[1:])
        elif expected_platform == "win32":
            expected_os = "win32"
        elif expected_platform.startswith("manylinux") is True:
            expected_os = "linux"
            if len(parts) >= 1 and parts[0] == "manylinux":
                if len(parts) >= 4:
                    expected_arch = "_".join(parts[3:])
            else:
                if len(parts) >= 2:
                    expected_arch = "_".join(parts[1:])
        elif expected_platform.startswith("musllinux_") is True:
            expected_os = "linux"
            if len(parts) >= 4:
                expected_arch = "_".join(parts[3:])
        elif expected_platform.startswith("linux_") is True:
            expected_os = "linux"
            if len(parts) >= 2:
                expected_arch = "_".join(parts[1:])

        if expected_os is not None:
            actual_os: str = sys.platform
            if actual_os != expected_os:
                _runtime_error(
                    "OS mismatch for this bundle.\n"
                    f"Expected: {expected_os} (from {expected_platform})\n"
                    f"Actual:   {actual_os}\n"
                )

        if expected_arch is not None:
            actual_machine: str = platform.machine()
            actual_arch: str = _normalize_arch(actual_machine)
            expected_arch_norm: str = _normalize_arch(expected_arch)
            if actual_arch != expected_arch_norm:
                _runtime_error(
                    "CPU architecture mismatch for this bundle.\n"
                    f"Expected: {expected_arch_norm} (from {expected_platform})\n"
                    f"Actual:   {actual_arch} (machine={actual_machine})\n"
                )


    def _cache_dir() -> pathlib.Path:
        """Return the extraction cache directory for this bundle.

        :returns: Cache directory path.
        """

        override: str | None = os.environ.get("PYTHON_FLATTENER_CACHE_DIR")
        if override is not None and len(override) > 0:
            base: pathlib.Path = pathlib.Path(override)
        else:
            base = pathlib.Path(tempfile.gettempdir()) / "python_flattener_cache"
        return base / _BUNDLE_SHA256


    def _safe_extract(zip_bytes: bytes, dest_dir: pathlib.Path) -> None:
        """Safely extract zip bytes into a destination directory.

        :param zip_bytes: Zip archive bytes.
        :param dest_dir: Destination directory.
        """

        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
            for info in zf.infolist():
                name: str = info.filename
                if "\\" in name:
                    _runtime_error(f"Refusing to extract backslash path: {name!r}")
                if ":" in name:
                    _runtime_error(f"Refusing to extract drive-like path: {name!r}")
                p = pathlib.PurePosixPath(name)
                if p.is_absolute() is True:
                    _runtime_error(f"Refusing to extract absolute path: {name!r}")
                if ".." in p.parts:
                    _runtime_error(f"Refusing to extract parent-traversal path: {name!r}")

                out_path: pathlib.Path = dest_dir.joinpath(*p.parts)
                if info.is_dir() is True:
                    out_path.mkdir(parents=True, exist_ok=True)
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, mode="r") as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)


    def _safe_extract_zipfile(*, zip_path: pathlib.Path, dest_dir: pathlib.Path) -> None:
        """Safely extract a zip file on disk into a destination directory.

        :param zip_path: Zip file path.
        :param dest_dir: Destination directory.
        """

        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, mode="r") as zf:
            for info in zf.infolist():
                name: str = info.filename
                if "\\" in name:
                    _runtime_error(f"Refusing to extract backslash path: {name!r}")
                if ":" in name:
                    _runtime_error(f"Refusing to extract drive-like path: {name!r}")
                p = pathlib.PurePosixPath(name)
                if p.is_absolute() is True:
                    _runtime_error(f"Refusing to extract absolute path: {name!r}")
                if ".." in p.parts:
                    _runtime_error(f"Refusing to extract parent-traversal path: {name!r}")

                out_path: pathlib.Path = dest_dir.joinpath(*p.parts)
                if info.is_dir() is True:
                    out_path.mkdir(parents=True, exist_ok=True)
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, mode="r") as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)


    def _ensure_extracted() -> pathlib.Path:
        """Ensure the bundle payload is extracted to disk.

        :returns: Extraction root directory.
        """

        root: pathlib.Path = _cache_dir()
        marker: pathlib.Path = root / ".extracted"
        if marker.is_file() is True:
            return root

        zip_bytes: bytes = base64.b64decode(_BUNDLE_B64.encode("ascii"))
        digest: str = hashlib.sha256(zip_bytes).hexdigest()
        if digest != _BUNDLE_SHA256:
            _runtime_error("Bundle payload checksum mismatch (corrupt file).\n")

        _safe_extract(zip_bytes=zip_bytes, dest_dir=root)
        marker.write_text("ok\n", encoding="utf-8")
        return root


    def _ensure_deps_ready(*, root: pathlib.Path) -> pathlib.Path:
        """Ensure the deps directory exists and is populated.

        :param root: Extraction root directory.
        :returns: deps directory path.
        """

        deps_dir: pathlib.Path = root / "deps"
        marker: pathlib.Path = root / ".deps_ready"
        if marker.is_file() is True and deps_dir.is_dir() is True:
            return deps_dir

        if deps_dir.is_dir() is True and marker.is_file() is False:
            return deps_dir

        layer_zip: pathlib.Path = root / "deps_layer.zip"
        if layer_zip.is_file() is False:
            deps_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok\n", encoding="utf-8")
            return deps_dir

        deps_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_zipfile(zip_path=layer_zip, dest_dir=deps_dir)
        marker.write_text("ok\n", encoding="utf-8")
        return deps_dir


    def _configure_sys_path(*, app_dir: pathlib.Path, deps_dir: pathlib.Path) -> None:
        """Configure ``sys.path`` so app + bundled deps are importable.

        :param app_dir: Extracted ``app/`` directory.
        :param deps_dir: Extracted ``deps/`` directory.
        """

        app_str: str = str(app_dir)
        if app_str in sys.path:
            sys.path.remove(app_str)
        sys.path.insert(0, app_str)

        deps_str: str = str(deps_dir)
        site.addsitedir(deps_str)
        if deps_str in sys.path:
            sys.path.remove(deps_str)
        sys.path.insert(1, deps_str)


    def _run(entry_root: pathlib.Path) -> None:
        """Execute the app entrypoint.

        :param entry_root: Extraction root directory.
        """

        app_dir: pathlib.Path = entry_root / "app"
        deps_dir: pathlib.Path = _ensure_deps_ready(root=entry_root)
        _configure_sys_path(app_dir=app_dir, deps_dir=deps_dir)

        kind: str = _ENTRY["kind"]
        if kind == "module":
            module_name: str = _ENTRY["module"]
            runpy.run_module(module_name, run_name="__main__", alter_sys=True)
            return
        if kind == "path":
            rel_path: str = _ENTRY["path"]
            entry_path: pathlib.Path = app_dir / rel_path
            if entry_path.is_file() is False:
                _runtime_error(f"Entry file missing after extraction: {rel_path!r}\n")
            sys.argv[0] = str(entry_path)
            runpy.run_path(str(entry_path), run_name="__main__")
            return

        _runtime_error(f"Invalid entry kind: {kind!r}\n")


    def _try_resolve_module_source(
        *, app_dir: pathlib.Path, module_name: str
    ) -> tuple[pathlib.Path, bool, pathlib.Path | None] | None:
        """Resolve a module name into a source file within the extracted app dir.

        :param app_dir: Extracted ``app/`` directory.
        :param module_name: Dotted module name.
        :returns: ``(source_path, is_package, package_dir)`` or ``None``.
        """

        parts: list[str] = module_name.split(".")
        pkg_dir: pathlib.Path = app_dir.joinpath(*parts)
        init_py: pathlib.Path = pkg_dir / "__init__.py"
        if init_py.is_file() is True:
            return (init_py, True, pkg_dir)

        mod_py: pathlib.Path = app_dir.joinpath(*parts).with_suffix(".py")
        if mod_py.is_file() is True:
            return (mod_py, False, None)

        return None


    def _resolve_import_source(*, app_dir: pathlib.Path) -> tuple[pathlib.Path, bool, pathlib.Path | None]:
        """Pick a source file to load into this module when imported.

        The preferred behavior is "transparent import": if the bundle file is named
        the same as the original module/package, ``import <name>`` should behave
        like importing the original module/package.

        :param app_dir: Extracted ``app/`` directory.
        :returns: ``(source_path, is_package, package_dir)``.
        """

        resolved_self = _try_resolve_module_source(app_dir=app_dir, module_name=__name__)
        if resolved_self is not None:
            return resolved_self

        kind: str = _ENTRY["kind"]
        if kind == "path":
            rel_path: str = _ENTRY["path"]
            entry_path: pathlib.Path = app_dir / rel_path
            if entry_path.is_file() is False:
                _runtime_error(f"Entry file missing after extraction: {rel_path!r}\n")
            return (entry_path, False, None)

        if kind == "module":
            module_name: str = _ENTRY["module"]
            resolved_entry = _try_resolve_module_source(
                app_dir=app_dir,
                module_name=module_name,
            )
            if resolved_entry is None:
                _runtime_error("Entry module missing after extraction.\n" f"module={module_name!r}\n")
            return resolved_entry

        _runtime_error(f"Invalid entry kind: {kind!r}\n")
        raise AssertionError("unreachable")


    def _exec_app_into_this_module(
        *,
        source_path: pathlib.Path,
        is_package: bool,
        package_dir: pathlib.Path | None,
    ) -> None:
        """Execute the extracted app module/package code into this module.

        :param source_path: Path to the extracted module source file.
        :param is_package: Whether this should behave like a package.
        :param package_dir: Package directory (required when ``is_package`` is true).
        """

        submodule_locations: list[str] | None = None
        if is_package is True:
            if package_dir is None:
                _runtime_error("Internal error: is_package true but package_dir missing.\n")
            submodule_locations = [str(package_dir)]

        spec = importlib.util.spec_from_file_location(
            __name__,
            str(source_path),
            submodule_search_locations=submodule_locations,
        )
        if spec is None or spec.loader is None:
            _runtime_error("Failed to create an import spec for extracted sources.\n")

        m = sys.modules[__name__]
        m.__spec__ = spec
        m.__loader__ = spec.loader
        m.__file__ = str(source_path)
        if is_package is True:
            if package_dir is None:
                _runtime_error("Internal error: package_dir missing.\n")
            m.__path__ = [str(package_dir)]
            m.__package__ = __name__
        else:
            m.__package__ = __name__.rpartition(".")[0]

        spec.loader.exec_module(m)


    _BOOTSTRAPPED_IMPORT: bool = False


    def _bootstrap_import() -> None:
        """Make importing the bundle behave like importing the original module/package."""

        global _BOOTSTRAPPED_IMPORT
        if _BOOTSTRAPPED_IMPORT is True:
            return
        _BOOTSTRAPPED_IMPORT = True

        _check_runtime_compat()
        root: pathlib.Path = _ensure_extracted()
        app_dir: pathlib.Path = root / "app"
        deps_dir: pathlib.Path = _ensure_deps_ready(root=root)
        _configure_sys_path(app_dir=app_dir, deps_dir=deps_dir)

        source_path: pathlib.Path
        is_package: bool
        package_dir: pathlib.Path | None
        source_path, is_package, package_dir = _resolve_import_source(app_dir=app_dir)
        _exec_app_into_this_module(
            source_path=source_path,
            is_package=is_package,
            package_dir=package_dir,
        )


    def main() -> None:
        """Program entrypoint."""

        _check_runtime_compat()
        root: pathlib.Path = _ensure_extracted()
        _run(entry_root=root)


    if __name__ == "__main__":
        main()
    else:
        _bootstrap_import()
    '''
).lstrip()
