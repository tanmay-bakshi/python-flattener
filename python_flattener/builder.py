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
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
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


def build_single_file(
    *,
    input_path: pathlib.Path,
    requirements_path: pathlib.Path,
    output_path: pathlib.Path,
    target: TargetConfig,
    entry_relpath: str | None,
    entry_module: str | None,
) -> None:
    """Build a single-file bundle.

    :param input_path: Python file or directory containing the app.
    :param requirements_path: requirements.txt path (can be empty).
    :param output_path: Output path for the bundled .py file.
    :param target: Target interpreter/platform config for wheel resolution.
    :param entry_relpath: Optional entry file path relative to the input directory.
    :param entry_module: Optional module name to run (python -m style).
    :raises BuildError: If bundling fails.
    """

    if input_path.exists() is False:
        raise BuildError(f"Input path does not exist: {input_path}")
    if requirements_path.exists() is False:
        raise BuildError(f"requirements.txt does not exist: {requirements_path}")
    if requirements_path.is_file() is False:
        raise BuildError(f"requirements path is not a file: {requirements_path}")

    if entry_relpath is not None and entry_module is not None:
        raise BuildError("Use at most one of --entry and --module.")

    with tempfile.TemporaryDirectory(prefix="python_flattener_build_") as td:
        build_root: pathlib.Path = pathlib.Path(td)
        wheel_dir: pathlib.Path = build_root / "wheels"
        staging_root: pathlib.Path = build_root / "staging"
        app_dir: pathlib.Path = staging_root / "app"
        deps_dir: pathlib.Path = staging_root / "deps"
        wheel_dir.mkdir(parents=True, exist_ok=True)
        staging_root.mkdir(parents=True, exist_ok=True)
        app_dir.mkdir(parents=True, exist_ok=True)
        deps_dir.mkdir(parents=True, exist_ok=True)

        app_layout: AppLayout = _stage_app(
            input_path=input_path,
            app_dir=app_dir,
            entry_relpath=entry_relpath,
            entry_module=entry_module,
        )

        wheel_files: list[pathlib.Path] = _download_wheels(
            requirements_path=requirements_path,
            wheel_dir=wheel_dir,
            target=target,
        )
        _install_wheels(wheel_files=wheel_files, deps_dir=deps_dir, target=target)

        bundle_zip_bytes: bytes = _zip_dir(staging_root)
        script_text: str = _render_single_file(
            bundle_zip_bytes=bundle_zip_bytes,
            entry=app_layout.entry,
            target=target,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(script_text, encoding="utf-8")


def _stage_app(
    *,
    input_path: pathlib.Path,
    app_dir: pathlib.Path,
    entry_relpath: str | None,
    entry_module: str | None,
) -> AppLayout:
    """Copy application sources into the staging directory and resolve entrypoint.

    :param input_path: File or directory input.
    :param app_dir: Destination staging ``app/`` directory.
    :param entry_relpath: Optional entry file path relative to input directory.
    :param entry_module: Optional module name to run.
    :returns: App layout information.
    :raises BuildError: If the app cannot be staged.
    """

    if input_path.is_file() is True:
        if entry_relpath is not None or entry_module is not None:
            raise BuildError("For file inputs, do not use --entry/--module; the file is the entry.")

        src_root: pathlib.Path = input_path.parent
        _copy_tree_app(src=src_root, dst=app_dir)

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
        _copy_tree_app(src=input_path, dst=staged_root)
        staged_entry_prefix = input_path.name
    else:
        staged_root = app_dir
        _copy_tree_app(src=input_path, dst=staged_root)
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


def _copy_tree_app(*, src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy an app directory tree to a destination, applying a conservative ignore list.

    :param src: Source directory.
    :param dst: Destination directory.
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
    }
    for p in src.rglob("*"):
        rel: pathlib.Path = p.relative_to(src)
        parts: tuple[str, ...] = rel.parts
        if len(parts) > 0 and parts[0] in ignore_names:
            continue

        if p.is_dir() is True:
            (dst / rel).mkdir(parents=True, exist_ok=True)
            continue

        if p.is_file() is True:
            if p.suffix in {".pyc", ".pyo"}:
                continue
            if p.name == ".DS_Store":
                continue
            target_path: pathlib.Path = dst / rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target_path)


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
    *, requirements_path: pathlib.Path, wheel_dir: pathlib.Path, target: TargetConfig
) -> list[pathlib.Path]:
    """Download wheels (and possibly sdists) for the given requirements.

    :param requirements_path: requirements.txt file.
    :param wheel_dir: Directory to download artifacts into.
    :param target: Target config for pip environment emulation.
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
            ]
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
            ]
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
            ]
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


def _pip(args: list[str]) -> None:
    """Invoke pip with the current interpreter.

    :param args: Arguments after ``-m pip``.
    :raises BuildError: If pip fails.
    """

    cmd: list[str] = [sys.executable, "-m", "pip", *args]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        out: str = proc.stdout
        err: str = proc.stderr
        joined: str = "\n".join(
            [
                "pip invocation failed:",
                f"cmd: {' '.join(cmd)}",
                f"exit: {proc.returncode}",
                "stdout:",
                out,
                "stderr:",
                err,
            ]
        )
        raise BuildError(joined)


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

        if expected_platform.startswith("macosx_") is True:
            expected_os = "darwin"
            expected_arch = expected_platform.split("_")[-1]
        elif expected_platform.startswith("win_") is True:
            expected_os = "win32"
            expected_arch = expected_platform.split("_")[-1]
        elif expected_platform == "win32":
            expected_os = "win32"
        elif (
            expected_platform.startswith("manylinux") is True
            or expected_platform.startswith("musllinux_") is True
            or expected_platform.startswith("linux_") is True
        ):
            expected_os = "linux"
            expected_arch = expected_platform.split("_")[-1]

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


    def _run(entry_root: pathlib.Path) -> None:
        """Execute the app entrypoint.

        :param entry_root: Extraction root directory.
        """

        app_dir: pathlib.Path = entry_root / "app"
        deps_dir: pathlib.Path = entry_root / "deps"
        sys.path.insert(0, str(app_dir))
        site.addsitedir(str(deps_dir))
        deps_str: str = str(deps_dir)
        if deps_str in sys.path:
            sys.path.remove(deps_str)
        sys.path.insert(1, deps_str)

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


    def main() -> None:
        """Program entrypoint."""

        _check_runtime_compat()
        root: pathlib.Path = _ensure_extracted()
        _run(entry_root=root)


    if __name__ == "__main__":
        main()
    '''
).lstrip()
