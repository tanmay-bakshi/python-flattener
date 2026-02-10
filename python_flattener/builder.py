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
    prefer_in_memory_runtime: bool = False,
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
    :param prefer_in_memory_runtime: Prefer an in-memory runtime in the generated script.
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
            prefer_in_memory_runtime=prefer_in_memory_runtime,
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
    prefer_in_memory_runtime: bool,
) -> None:
    """Write the final self-extracting bundle script to disk.

    This avoids constructing a massive in-memory string for the base64 payload.

    :param output_path: Output path for the bundled .py file.
    :param bundle_zip_bytes: Zipped payload bytes (outer payload zip).
    :param entry: Entrypoint information.
    :param target: Target config embedded for runtime validation.
    :param logger: Logger for progress output.
    :param prefer_in_memory_runtime: Prefer an in-memory runtime in the generated script.
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
    prefer_text: str = "True" if prefer_in_memory_runtime is True else "False"
    runtime = runtime.replace("__PYFLAT_PREFER_IN_MEMORY__", prefer_text)

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
    import builtins
    import ctypes
    from dataclasses import dataclass
    import hashlib
    import importlib.abc
    import importlib.machinery
    import importlib.resources
    import importlib.util
    import io
    import os
    import pathlib
    import platform
    import runpy
    import shutil
    import site
    import struct
    import sys
    import tempfile
    import zipfile


    _BUNDLE_SHA256: str = "__PYFLAT_SHA256__"
    _TARGET: dict[str, str] = __PYFLAT_TARGET_JSON__
    _ENTRY: dict[str, str] = __PYFLAT_ENTRY_JSON__
    _PREFER_IN_MEMORY: bool = __PYFLAT_PREFER_IN_MEMORY__

    _BUNDLE_B64: str = r"""__PYFLAT_BUNDLE_B64__"""

    _MEM_SETUP_DONE: bool = False
    _MEM_FDS: list[int] = []
    _MEM_LIB_HANDLES: list[ctypes.CDLL] = []
    _MEM_PAYLOAD_ZIP_PATH: str | None = None
    _MEM_DEPS_ZIP_PATH: str | None = None
    _MEM_EXTENSION_MODULE_PATHS: dict[str, str] = {}
    _OPEN_PATCHED: bool = False
    _ORIG_OPEN = builtins.open
    _RESOURCES_PATCHED: bool = False
    _ORIG_IMPORTLIB_AS_FILE = importlib.resources.as_file


    def _runtime_error(message: str) -> None:
        """Exit with a message.

        :param message: Error message.
        """

        sys.stderr.write(message)
        if message.endswith("\n") is False:
            sys.stderr.write("\n")
        raise SystemExit(2)


    def _parse_env_bool(value: str) -> bool | None:
        """Parse a string into a boolean.

        :param value: Raw environment variable string.
        :returns: Parsed boolean, or ``None`` if unknown.
        """

        v: str = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
        return None


    def _use_in_memory_runtime() -> bool:
        """Return whether the in-memory runtime mode should be used.

        This can be overridden via ``PYTHON_FLATTENER_IN_MEMORY``.

        :returns: ``True`` if in-memory mode should be enabled.
        """

        override: str | None = os.environ.get("PYTHON_FLATTENER_IN_MEMORY")
        if override is not None and len(override) > 0:
            parsed: bool | None = _parse_env_bool(override)
            if parsed is not None:
                return parsed

        return _PREFER_IN_MEMORY


    def _bundle_zip_bytes() -> bytes:
        """Decode and validate the embedded bundle payload.

        :returns: Bundle zip bytes.
        """

        zip_bytes: bytes = base64.b64decode(_BUNDLE_B64.encode("ascii"))
        digest: str = hashlib.sha256(zip_bytes).hexdigest()
        if digest != _BUNDLE_SHA256:
            _runtime_error("Bundle payload checksum mismatch (corrupt file).\n")
        return zip_bytes


    def _proc_fd_path(fd: int) -> str:
        """Build a ``/proc`` path for a file descriptor.

        :param fd: File descriptor number.
        :returns: Path like ``/proc/self/fd/<fd>``.
        """

        return f"/proc/self/fd/{fd}"


    def _memfd_from_bytes(*, name: str, data: bytes) -> int:
        """Create a Linux memfd containing the given bytes.

        :param name: memfd name (debug-only).
        :param data: Data to write.
        :returns: File descriptor number.
        """

        if hasattr(os, "memfd_create") is False:
            _runtime_error("In-memory mode requires Linux with os.memfd_create().\n")

        flags: int = 0
        if hasattr(os, "MFD_CLOEXEC") is True:
            flags = os.MFD_CLOEXEC

        fd: int = os.memfd_create(name, flags=flags)
        os.set_inheritable(fd, False)
        _MEM_FDS.append(fd)

        view = memoryview(data)
        off: int = 0
        total: int = len(view)
        while off < total:
            n_written: int = os.write(fd, view[off:])
            if n_written <= 0:
                _runtime_error("Failed to write bundle bytes into memfd.\n")
            off += n_written

        os.lseek(fd, 0, os.SEEK_SET)
        return fd


    @dataclass(frozen=True, slots=True)
    class _SoInfo:
        """Shared object metadata for the in-memory runtime.

        :ivar member: Zip member name within the deps layer zip.
        :ivar fd_path: ``/proc/self/fd/<fd>`` path for the memfd holding this object.
        :ivar soname: Library SONAME (or basename fallback).
        :ivar needed: DT_NEEDED dependency names.
        :ivar module_name: Dotted module name if this is a Python extension module.
        """

        member: str
        fd_path: str
        soname: str
        needed: tuple[str, ...]
        module_name: str | None


    def _is_shared_object_member(member: str) -> bool:
        """Check whether a zip member name looks like a shared object.

        :param member: Zip member name.
        :returns: ``True`` if it looks like a shared object file.
        """

        name: str = member.lower()
        if name.endswith(".so") is True:
            return True
        if ".so." in name:
            return True
        return False


    def _extension_module_name_from_member(member: str) -> str | None:
        """Convert a deps-layer zip member into an extension module name.

        This only returns a name when all path segments are valid Python identifiers.

        :param member: Zip member name (POSIX path).
        :returns: Dotted module name, or ``None``.
        """

        parts: list[str] = member.split("/")
        if len(parts) == 0:
            return None

        filename: str = parts[-1]
        base: str = filename.split(".", 1)[0]
        if base == "__init__":
            return None
        if base.isidentifier() is False:
            return None

        dir_parts: list[str] = parts[0:-1]
        for p in dir_parts:
            if len(p) == 0:
                return None
            if p.isidentifier() is False:
                return None

        if len(dir_parts) == 0:
            return base
        return ".".join([*dir_parts, base])


    def _elf_soname_and_needed(data: bytes) -> tuple[str | None, tuple[str, ...]]:
        """Extract ELF SONAME and DT_NEEDED entries from a shared object.

        This supports 64-bit little-endian ELF files (common for manylinux x86_64/aarch64).

        :param data: ELF file bytes.
        :returns: ``(soname, needed)``.
        """

        if len(data) < 64:
            return (None, ())
        if data[0:4] != b"\x7fELF":
            return (None, ())

        ei_class: int = data[4]
        ei_data: int = data[5]
        if ei_class != 2 or ei_data != 1:
            return (None, ())

        try:
            e_phoff: int = struct.unpack_from("<Q", data, 32)[0]
            e_phentsize: int = struct.unpack_from("<H", data, 54)[0]
            e_phnum: int = struct.unpack_from("<H", data, 56)[0]
        except struct.error:
            return (None, ())

        load_segs: list[tuple[int, int, int]] = []
        dyn_off: int | None = None
        dyn_size: int | None = None

        i: int = 0
        while i < e_phnum:
            ph_base: int = e_phoff + i * e_phentsize
            try:
                p_type: int = struct.unpack_from("<I", data, ph_base)[0]
            except struct.error:
                break

            if p_type == 1 or p_type == 2:
                try:
                    p_offset: int = struct.unpack_from("<Q", data, ph_base + 8)[0]
                    p_vaddr: int = struct.unpack_from("<Q", data, ph_base + 16)[0]
                    p_filesz: int = struct.unpack_from("<Q", data, ph_base + 32)[0]
                except struct.error:
                    break

                if p_type == 1:
                    load_segs.append((p_vaddr, p_filesz, p_offset))
                else:
                    dyn_off = p_offset
                    dyn_size = p_filesz

            i += 1

        if dyn_off is None or dyn_size is None:
            return (None, ())

        dt_strtab: int | None = None
        dt_strsz: int | None = None
        dt_soname_off: int | None = None
        needed_offs: list[int] = []

        dyn_end: int = dyn_off + dyn_size
        pos: int = dyn_off
        while pos + 16 <= dyn_end:
            try:
                d_tag: int = struct.unpack_from("<q", data, pos)[0]
                d_val: int = struct.unpack_from("<Q", data, pos + 8)[0]
            except struct.error:
                break

            if d_tag == 0:
                break
            if d_tag == 1:
                needed_offs.append(int(d_val))
            elif d_tag == 5:
                dt_strtab = int(d_val)
            elif d_tag == 10:
                dt_strsz = int(d_val)
            elif d_tag == 14:
                dt_soname_off = int(d_val)

            pos += 16

        if dt_strtab is None or dt_strsz is None:
            return (None, ())

        strtab_off: int | None = None
        for vaddr, filesz, off0 in load_segs:
            if dt_strtab >= vaddr and dt_strtab < vaddr + filesz:
                strtab_off = off0 + (dt_strtab - vaddr)
                break

        if strtab_off is None:
            return (None, ())
        if strtab_off < 0:
            return (None, ())

        strtab_end: int = strtab_off + dt_strsz
        if strtab_end > len(data):
            strtab_end = len(data)
        strtab: bytes = data[strtab_off:strtab_end]

        def read_cstr(off: int) -> str:
            if off < 0 or off >= len(strtab):
                return ""
            end: int = strtab.find(b"\x00", off)
            if end < 0:
                end = len(strtab)
            try:
                return strtab[off:end].decode("utf-8")
            except UnicodeDecodeError:
                return strtab[off:end].decode("utf-8", errors="replace")

        soname: str | None = None
        if dt_soname_off is not None:
            s: str = read_cstr(dt_soname_off)
            if len(s) > 0:
                soname = s

        needed: list[str] = []
        for off_needed in needed_offs:
            s2: str = read_cstr(off_needed)
            if len(s2) > 0:
                needed.append(s2)

        return (soname, tuple(needed))


    def _collect_shared_objects_from_deps_zip(*, deps_zip_path: str) -> list[_SoInfo]:
        """Collect shared object metadata from a deps-layer zip.

        :param deps_zip_path: Path to the deps-layer zip (via memfd).
        :returns: Shared object metadata list.
        """

        infos: list[_SoInfo] = []
        with zipfile.ZipFile(deps_zip_path, mode="r") as zf:
            for zi in zf.infolist():
                if zi.is_dir() is True:
                    continue
                member: str = zi.filename
                if _is_shared_object_member(member) is False:
                    continue

                data: bytes = zf.read(member)
                base_name: str = pathlib.PurePosixPath(member).name
                fd: int = _memfd_from_bytes(name=base_name[0:60], data=data)
                fd_path: str = _proc_fd_path(fd)

                soname_raw: str | None
                needed: tuple[str, ...]
                soname_raw, needed = _elf_soname_and_needed(data)
                soname: str = soname_raw if soname_raw is not None else base_name
                module_name: str | None = _extension_module_name_from_member(member)

                infos.append(
                    _SoInfo(
                        member=member,
                        fd_path=fd_path,
                        soname=soname,
                        needed=needed,
                        module_name=module_name,
                    )
                )

        return infos


    def _topo_sort_shared_objects(infos: list[_SoInfo]) -> list[_SoInfo]:
        """Topologically sort shared objects by DT_NEEDED dependencies.

        :param infos: Shared object infos.
        :returns: Sorted list (best-effort).
        """

        index_by_soname: dict[str, int] = {}
        i: int = 0
        while i < len(infos):
            so: _SoInfo = infos[i]
            if so.soname not in index_by_soname:
                index_by_soname[so.soname] = i
            i += 1

        indegree: list[int] = [0 for _ in infos]
        edges: dict[int, set[int]] = {}
        for j, so2 in enumerate(infos):
            for needed in so2.needed:
                dep_idx: int | None = index_by_soname.get(needed)
                if dep_idx is None:
                    continue
                if dep_idx == j:
                    continue
                if dep_idx not in edges:
                    edges[dep_idx] = set()
                if j not in edges[dep_idx]:
                    edges[dep_idx].add(j)
                    indegree[j] += 1

        ready: list[int] = [k for k, d in enumerate(indegree) if d == 0]
        ready.sort(key=lambda idx: infos[idx].soname)
        order: list[int] = []
        while len(ready) > 0:
            idx0: int = ready.pop(0)
            order.append(idx0)

            nexts: set[int] = edges.get(idx0, set())
            for dep in sorted(nexts, key=lambda x: infos[x].soname):
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    ready.append(dep)
                    ready.sort(key=lambda x: infos[x].soname)

        if len(order) != len(infos):
            seen: set[int] = set(order)
            remaining: list[int] = [k for k in range(len(infos)) if k not in seen]
            remaining.sort(key=lambda x: infos[x].soname)
            order.extend(remaining)

        return [infos[k] for k in order]


    def _preload_shared_objects(infos: list[_SoInfo]) -> None:
        """Pre-load shared objects into the process.

        This helps resolve wheels that rely on ``$ORIGIN`` RPATHs, by ensuring needed
        libraries are already loaded globally before extension modules import.

        :param infos: Shared object infos.
        """

        if len(infos) == 0:
            return

        rtld_global: int = os.RTLD_GLOBAL if hasattr(os, "RTLD_GLOBAL") is True else 0
        rtld_now: int = os.RTLD_NOW if hasattr(os, "RTLD_NOW") is True else 0
        sys.setdlopenflags(sys.getdlopenflags() | rtld_global)

        ordered: list[_SoInfo] = _topo_sort_shared_objects(infos)
        pending: list[_SoInfo] = ordered
        max_passes: int = 8
        pass_idx: int = 0

        while pass_idx < max_passes and len(pending) > 0:
            next_pending: list[_SoInfo] = []
            loaded_count: int = 0
            for so in pending:
                try:
                    h = ctypes.CDLL(so.fd_path, mode=rtld_global | rtld_now)
                except OSError:
                    next_pending.append(so)
                    continue
                _MEM_LIB_HANDLES.append(h)
                loaded_count += 1

            pending = next_pending
            if loaded_count == 0:
                break
            pass_idx += 1

        if len(pending) > 0:
            lines: list[str] = ["In-memory runtime failed to preload some shared libraries:"]
            for so in pending[0:10]:
                lines.append(f"- {so.member}")
            if len(pending) > 10:
                lines.append(f"... and {len(pending) - 10} more")
            _runtime_error("\n".join(lines) + "\n")


    class _MemExtensionFinder(importlib.abc.MetaPathFinder):
        """Meta path finder for extension modules stored in memfd."""

        _module_paths: dict[str, str]

        def __init__(self, module_paths: dict[str, str]) -> None:
            """Initialize the finder.

            :param module_paths: Mapping from dotted module name to memfd path.
            """

            self._module_paths = module_paths

        def find_spec(  # type: ignore[override]
            self,
            fullname: str,
            path: object | None,
            target: object | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            """Find an extension module spec by name.

            :param fullname: Module name being imported.
            :param path: Package path hint (unused).
            :param target: Target module (unused).
            :returns: A module spec if this is a bundled extension module.
            """

            origin: str | None = self._module_paths.get(fullname)
            if origin is None:
                return None

            loader = importlib.machinery.ExtensionFileLoader(fullname, origin)
            return importlib.util.spec_from_file_location(fullname, origin, loader=loader)


    def _install_extension_finder(*, module_paths: dict[str, str]) -> None:
        """Install the in-memory extension module finder.

        :param module_paths: Mapping of module name to memfd path.
        """

        if len(module_paths) == 0:
            return
        sys.meta_path.insert(0, _MemExtensionFinder(module_paths))


    def _is_valid_pkg_segment(segment: str) -> bool:
        """Return whether a path segment can be used as a package name.

        :param segment: Path segment.
        :returns: ``True`` if it can be used as a Python package name.
        """

        if len(segment) == 0:
            return False
        if segment == "__pycache__":
            return False
        return segment.isidentifier()


    def _discover_namespace_packages_in_zip(
        *, zip_path: str, root_prefix: str, base_path: str
    ) -> dict[str, list[str]]:
        """Discover namespace packages in a zip payload.

        Python supports namespace packages (PEP 420) for directories on ``sys.path``
        without an ``__init__.py``. When running from zip paths, the default import
        machinery may fail to recognize namespace packages like ``google``.

        :param zip_path: Zip file path (via memfd).
        :param root_prefix: Member prefix that acts as the import root (e.g. ``"app/"``).
        :param base_path: ``sys.path`` entry corresponding to ``root_prefix``.
        :returns: Mapping of namespace fullname to submodule search locations.
        """

        init_dirs: set[str] = set()
        module_fullnames: set[str] = set()
        candidate_dirs: set[str] = set()

        with zipfile.ZipFile(zip_path, mode="r") as zf:
            for zi in zf.infolist():
                if zi.is_dir() is True:
                    continue

                member: str = zi.filename
                rel: str
                if len(root_prefix) > 0:
                    if member.startswith(root_prefix) is False:
                        continue
                    rel = member[len(root_prefix) :]
                else:
                    rel = member

                if len(rel) == 0:
                    continue

                parts: list[str] = rel.split("/")
                if len(parts) == 0:
                    continue

                filename: str = parts[-1]
                dir_parts: list[str] = parts[0:-1]

                acc: list[str] = []
                for p in dir_parts:
                    acc.append(p)
                    candidate_dirs.add("/".join(acc))

                if filename == "__init__.py":
                    init_dirs.add("/".join(dir_parts))
                    continue

                # Track "real" modules so the namespace finder does not mask them.
                if filename.endswith(".py") is True:
                    base: str = filename[0:-3]
                    if base == "__init__":
                        continue
                    if base.isidentifier() is False:
                        continue

                    ok_dirs: bool = True
                    for d in dir_parts:
                        if _is_valid_pkg_segment(d) is False:
                            ok_dirs = False
                            break
                    if ok_dirs is False:
                        continue

                    fullname: str
                    if len(dir_parts) == 0:
                        fullname = base
                    else:
                        fullname = ".".join([*dir_parts, base])
                    module_fullnames.add(fullname)
                    continue

                if _is_shared_object_member(rel) is True:
                    ext_name: str | None = _extension_module_name_from_member(rel)
                    if ext_name is not None:
                        module_fullnames.add(ext_name)

        ns_paths: dict[str, list[str]] = {}
        for dir_path in sorted(candidate_dirs):
            if dir_path in init_dirs:
                continue

            segs: list[str] = dir_path.split("/")
            ok: bool = True
            for seg in segs:
                if _is_valid_pkg_segment(seg) is False:
                    ok = False
                    break
            if ok is False:
                continue

            fullname2: str = ".".join(segs)
            if fullname2 in module_fullnames:
                continue

            ns_paths[fullname2] = [f"{base_path}/{dir_path}"]

        return ns_paths


    def _merge_namespace_paths(
        *, base: dict[str, list[str]], extra: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Merge two namespace package mappings.

        :param base: Base mapping (modified in-place).
        :param extra: Extra mapping to merge in.
        :returns: The merged mapping.
        """

        for fullname, locations in extra.items():
            if fullname in base:
                dest: list[str] = base[fullname]
                for loc in locations:
                    if loc not in dest:
                        dest.append(loc)
                continue

            base[fullname] = list(locations)

        return base


    class _MemNamespaceFinder(importlib.abc.MetaPathFinder):
        """Meta path finder for namespace packages inside in-memory zip paths."""

        _ns_paths: dict[str, list[str]]

        def __init__(self, ns_paths: dict[str, list[str]]) -> None:
            """Initialize the finder.

            :param ns_paths: Mapping from namespace fullname to search locations.
            """

            self._ns_paths = ns_paths

        def find_spec(  # type: ignore[override]
            self,
            fullname: str,
            path: object | None,
            target: object | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            """Find a namespace package spec by name.

            :param fullname: Module name being imported.
            :param path: Package path hint (unused).
            :param target: Target module (unused).
            :returns: A module spec if this is a bundled namespace package.
            """

            locations: list[str] | None = self._ns_paths.get(fullname)
            if locations is None:
                return None

            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = list(locations)
            return spec


    def _install_namespace_finder(*, ns_paths: dict[str, list[str]]) -> None:
        """Install the in-memory namespace package finder.

        :param ns_paths: Mapping from namespace fullname to search locations.
        """

        if len(ns_paths) == 0:
            return
        sys.meta_path.insert(0, _MemNamespaceFinder(ns_paths))


    def _patch_open_for_zip_paths(*, payload_zip_path: str, deps_zip_path: str | None) -> None:
        """Patch ``open()`` so zip-internal pseudo-paths can be read without extraction.

        Some third-party packages incorrectly do ``open(str(importlib.resources.files(...)))``.
        When running from a zip path, that string is not a real filesystem path. This hook
        intercepts such reads and serves the bytes from the bundled zip payloads.

        :param payload_zip_path: Path to the outer bundle zip (via memfd).
        :param deps_zip_path: Optional path to the deps-layer zip (via memfd).
        """

        global _OPEN_PATCHED
        if _OPEN_PATCHED is True:
            return

        zip_cache: dict[str, zipfile.ZipFile] = {}

        def read_member(zip_path: str, member: str) -> bytes:
            zf: zipfile.ZipFile | None = zip_cache.get(zip_path)
            if zf is None:
                zf = zipfile.ZipFile(zip_path, mode="r")
                zip_cache[zip_path] = zf
            return zf.read(member)

        payload_prefix: str = payload_zip_path + "/"
        deps_prefix: str | None = None
        if deps_zip_path is not None:
            deps_prefix = deps_zip_path + "/"

        def open_compat(  # type: ignore[override]
            file: object,
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
            closefd: bool = True,
            opener: object | None = None,
        ) -> io.IOBase:
            """Open a file path, with support for bundled zip pseudo-paths.

            :param file: File path or file descriptor.
            :param mode: Open mode.
            :returns: An IO object.
            """

            write_like: bool = False
            for ch in ("w", "a", "x", "+"):
                if ch in mode:
                    write_like = True
                    break
            if write_like is True:
                return _ORIG_OPEN(file, mode, buffering, encoding, errors, newline, closefd, opener)  # type: ignore[return-value]

            try:
                path0 = os.fspath(file)
            except TypeError:
                return _ORIG_OPEN(file, mode, buffering, encoding, errors, newline, closefd, opener)  # type: ignore[return-value]

            if isinstance(path0, str) is False:
                return _ORIG_OPEN(file, mode, buffering, encoding, errors, newline, closefd, opener)  # type: ignore[return-value]

            data: bytes | None = None
            if path0.startswith(payload_prefix) is True:
                member0: str = path0[len(payload_prefix) :]
                try:
                    data = read_member(payload_zip_path, member0)
                except KeyError:
                    data = None
            elif deps_prefix is not None and path0.startswith(deps_prefix) is True:
                member1: str = path0[len(deps_prefix) :]
                try:
                    data = read_member(deps_zip_path, member1)  # type: ignore[arg-type]
                except KeyError:
                    data = None

            if data is None:
                return _ORIG_OPEN(file, mode, buffering, encoding, errors, newline, closefd, opener)  # type: ignore[return-value]

            bio = io.BytesIO(data)
            if "b" in mode:
                return bio

            enc: str = encoding if encoding is not None else "utf-8"
            err: str = errors if errors is not None else "strict"
            return io.TextIOWrapper(bio, encoding=enc, errors=err, newline=newline)

        builtins.open = open_compat  # type: ignore[assignment]
        _OPEN_PATCHED = True


    class _MemfdResourceFile:
        """Context manager that exposes a zip resource as a memfd-backed ``Path``."""

        _traversable: zipfile.Path
        _fd: int | None

        def __init__(self, traversable: zipfile.Path) -> None:
            """Initialize the context manager.

            :param traversable: Zipfile traversable for a single file.
            """

            self._traversable = traversable
            self._fd = None

        def __enter__(self) -> pathlib.Path:
            """Enter the context manager.

            :returns: A real filesystem path (via ``/proc/self/fd/<fd>``).
            """

            if self._traversable.is_dir() is True:
                _runtime_error("In-memory mode does not support importlib.resources.as_file() for directories.\n")

            data: bytes = self._traversable.read_bytes()
            fd: int = _memfd_from_bytes(name="pyflat_resource", data=data)
            self._fd = fd
            return pathlib.Path(_proc_fd_path(fd))

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            """Exit the context manager.

            :param exc_type: Exception type (unused).
            :param exc: Exception instance (unused).
            :param tb: Traceback (unused).
            :returns: ``False`` to propagate exceptions.
            """

            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
            return False


    def _patch_importlib_resources_for_memfd() -> None:
        """Patch ``importlib.resources.as_file`` to avoid temp files in in-memory mode."""

        global _RESOURCES_PATCHED
        if _RESOURCES_PATCHED is True:
            return

        def as_file_compat(traversable: object) -> object:
            if isinstance(traversable, zipfile.Path) is True:
                return _MemfdResourceFile(traversable)
            return _ORIG_IMPORTLIB_AS_FILE(traversable)

        importlib.resources.as_file = as_file_compat  # type: ignore[assignment]
        _RESOURCES_PATCHED = True


    def _configure_sys_path_in_memory(*, payload_zip_path: str, deps_zip_path: str | None) -> None:
        """Configure ``sys.path`` for the in-memory runtime mode.

        :param payload_zip_path: Path to the outer bundle zip (via memfd).
        :param deps_zip_path: Optional path to the deps-layer zip (via memfd).
        """

        app_path: str = f"{payload_zip_path}/app"
        if app_path in sys.path:
            sys.path.remove(app_path)
        sys.path.insert(0, app_path)

        if deps_zip_path is None:
            return

        if deps_zip_path in sys.path:
            sys.path.remove(deps_zip_path)
        sys.path.insert(1, deps_zip_path)


    def _ensure_in_memory_setup() -> tuple[str, str | None]:
        """Initialize the in-memory runtime mode if needed.

        :returns: ``(payload_zip_path, deps_zip_path)``.
        """

        global _MEM_SETUP_DONE
        global _MEM_PAYLOAD_ZIP_PATH
        global _MEM_DEPS_ZIP_PATH
        global _MEM_EXTENSION_MODULE_PATHS

        if _MEM_SETUP_DONE is True:
            if _MEM_PAYLOAD_ZIP_PATH is None:
                _runtime_error("Internal error: in-memory setup marked done but payload missing.\n")
            return (_MEM_PAYLOAD_ZIP_PATH, _MEM_DEPS_ZIP_PATH)

        sys.dont_write_bytecode = True

        zip_bytes: bytes = _bundle_zip_bytes()
        payload_fd: int = _memfd_from_bytes(name="pyflat_bundle", data=zip_bytes)
        payload_zip_path: str = _proc_fd_path(payload_fd)
        _MEM_PAYLOAD_ZIP_PATH = payload_zip_path

        deps_zip_path: str | None = None
        try:
            with zipfile.ZipFile(payload_zip_path, mode="r") as zf:
                deps_bytes: bytes = zf.read("deps_layer.zip")
        except KeyError:
            deps_bytes = b""

        if len(deps_bytes) > 0:
            deps_fd: int = _memfd_from_bytes(name="pyflat_deps", data=deps_bytes)
            deps_zip_path = _proc_fd_path(deps_fd)
            _MEM_DEPS_ZIP_PATH = deps_zip_path

            infos: list[_SoInfo] = _collect_shared_objects_from_deps_zip(deps_zip_path=deps_zip_path)
            module_paths: dict[str, str] = {}
            for so in infos:
                if so.module_name is not None:
                    module_paths[so.module_name] = so.fd_path
            _MEM_EXTENSION_MODULE_PATHS = module_paths

            _preload_shared_objects(infos)
            _install_extension_finder(module_paths=module_paths)

        ns_paths: dict[str, list[str]] = {}
        app_path: str = f"{payload_zip_path}/app"
        app_ns: dict[str, list[str]] = _discover_namespace_packages_in_zip(
            zip_path=payload_zip_path,
            root_prefix="app/",
            base_path=app_path,
        )
        _merge_namespace_paths(base=ns_paths, extra=app_ns)

        if deps_zip_path is not None:
            deps_ns: dict[str, list[str]] = _discover_namespace_packages_in_zip(
                zip_path=deps_zip_path,
                root_prefix="",
                base_path=deps_zip_path,
            )
            _merge_namespace_paths(base=ns_paths, extra=deps_ns)

        _install_namespace_finder(ns_paths=ns_paths)

        _configure_sys_path_in_memory(payload_zip_path=payload_zip_path, deps_zip_path=deps_zip_path)
        _patch_open_for_zip_paths(payload_zip_path=payload_zip_path, deps_zip_path=deps_zip_path)
        _patch_importlib_resources_for_memfd()
        _MEM_SETUP_DONE = True
        return (payload_zip_path, deps_zip_path)


    def _read_zip_member(*, zip_path: str, member: str) -> bytes:
        """Read a member from a zip file at ``zip_path``.

        :param zip_path: Zip path (via memfd).
        :param member: Member name to read.
        :returns: Member bytes.
        """

        with zipfile.ZipFile(zip_path, mode="r") as zf:
            try:
                return zf.read(member)
            except KeyError:
                _runtime_error(f"Missing bundled file: {member!r}\n")
                raise AssertionError("unreachable")


    def _try_resolve_module_source_in_memory(
        *, payload_zip_path: str, module_name: str
    ) -> tuple[str, bool, str | None] | None:
        """Resolve a module name into a source file inside the bundled app.

        :param payload_zip_path: Path to the outer bundle zip (via memfd).
        :param module_name: Dotted module name.
        :returns: ``(source_member, is_package, package_dir_member)`` or ``None``.
        """

        parts: list[str] = module_name.split(".")
        pkg_member: str = "app/" + "/".join(parts) + "/__init__.py"
        mod_member: str = "app/" + "/".join(parts) + ".py"

        with zipfile.ZipFile(payload_zip_path, mode="r") as zf:
            try:
                zf.getinfo(pkg_member)
                pkg_dir_member: str = "app/" + "/".join(parts)
                return (pkg_member, True, pkg_dir_member)
            except KeyError:
                pass

            try:
                zf.getinfo(mod_member)
                return (mod_member, False, None)
            except KeyError:
                return None


    def _resolve_import_source_in_memory(
        *, payload_zip_path: str
    ) -> tuple[str, bool, str | None]:
        """Pick a source file to load into this module when imported (in-memory mode).

        :param payload_zip_path: Path to the outer bundle zip (via memfd).
        :returns: ``(source_member, is_package, package_dir_member)``.
        """

        resolved_self = _try_resolve_module_source_in_memory(
            payload_zip_path=payload_zip_path,
            module_name=__name__,
        )
        if resolved_self is not None:
            return resolved_self

        kind: str = _ENTRY["kind"]
        if kind == "path":
            rel_path: str = _ENTRY["path"]
            member: str = f"app/{rel_path}"
            with zipfile.ZipFile(payload_zip_path, mode="r") as zf:
                try:
                    zf.getinfo(member)
                except KeyError:
                    _runtime_error(f"Entry file missing in bundle: {rel_path!r}\n")
            return (member, False, None)

        if kind == "module":
            module_name: str = _ENTRY["module"]
            resolved_entry = _try_resolve_module_source_in_memory(
                payload_zip_path=payload_zip_path,
                module_name=module_name,
            )
            if resolved_entry is None:
                _runtime_error("Entry module missing in bundle.\n" f"module={module_name!r}\n")
            return resolved_entry

        _runtime_error(f"Invalid entry kind: {kind!r}\n")
        raise AssertionError("unreachable")


    def _exec_source_into_this_module_in_memory(
        *,
        payload_zip_path: str,
        source_member: str,
        is_package: bool,
        package_dir_member: str | None,
    ) -> None:
        """Execute bundled app code into this module object (in-memory mode).

        :param payload_zip_path: Path to the outer bundle zip (via memfd).
        :param source_member: Zip member for the source file (e.g. ``app/tools.py``).
        :param is_package: Whether this should behave like a package.
        :param package_dir_member: Package directory member (required for packages).
        """

        if __name__ not in sys.modules:
            _runtime_error("Internal error: importing module but missing from sys.modules.\n")
        m = sys.modules[__name__]

        file_path: str = f"{payload_zip_path}/{source_member}"
        m.__file__ = file_path
        if is_package is True:
            if package_dir_member is None:
                _runtime_error("Internal error: is_package true but package_dir_member missing.\n")
            m.__path__ = [f"{payload_zip_path}/{package_dir_member}"]
            m.__package__ = __name__
        else:
            m.__package__ = __name__.rpartition(".")[0]

        src_bytes: bytes = _read_zip_member(zip_path=payload_zip_path, member=source_member)
        try:
            src_text: str = src_bytes.decode("utf-8")
        except UnicodeDecodeError:
            _runtime_error(f"Failed to decode bundled source as UTF-8: {source_member!r}\n")

        code = compile(src_text, file_path, "exec")
        exec(code, m.__dict__)


    def _bootstrap_import_in_memory() -> None:
        """Make importing the bundle behave like importing the original module/package (in-memory)."""

        payload_zip_path: str
        deps_zip_path: str | None
        payload_zip_path, deps_zip_path = _ensure_in_memory_setup()

        source_member: str
        is_package: bool
        package_dir_member: str | None
        source_member, is_package, package_dir_member = _resolve_import_source_in_memory(
            payload_zip_path=payload_zip_path
        )
        _exec_source_into_this_module_in_memory(
            payload_zip_path=payload_zip_path,
            source_member=source_member,
            is_package=is_package,
            package_dir_member=package_dir_member,
        )


    def _run_in_memory() -> None:
        """Execute the app entrypoint (in-memory mode)."""

        payload_zip_path: str
        deps_zip_path: str | None
        payload_zip_path, deps_zip_path = _ensure_in_memory_setup()

        kind: str = _ENTRY["kind"]
        if kind == "module":
            module_name: str = _ENTRY["module"]
            runpy.run_module(module_name, run_name="__main__", alter_sys=True)
            return
        if kind == "path":
            rel_path: str = _ENTRY["path"]
            member: str = f"app/{rel_path}"
            src_bytes: bytes = _read_zip_member(zip_path=payload_zip_path, member=member)
            try:
                src_text: str = src_bytes.decode("utf-8")
            except UnicodeDecodeError:
                _runtime_error(f"Failed to decode bundled source as UTF-8: {member!r}\n")

            file_path: str = f"{payload_zip_path}/{member}"
            sys.argv[0] = file_path
            g: dict[str, object] = {
                "__name__": "__main__",
                "__file__": file_path,
                "__package__": "",
                "__builtins__": __builtins__,
            }
            code = compile(src_text, file_path, "exec")
            exec(code, g)
            return

        _runtime_error(f"Invalid entry kind: {kind!r}\n")


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
        if _use_in_memory_runtime() is True:
            _bootstrap_import_in_memory()
            return
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
        if _use_in_memory_runtime() is True:
            _run_in_memory()
            return
        root: pathlib.Path = _ensure_extracted()
        _run(entry_root=root)


    if __name__ == "__main__":
        main()
    else:
        _bootstrap_import()
    '''
).lstrip()
