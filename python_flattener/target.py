"""Target resolution helpers.

This module is intentionally small and "pragmatic":

- It accepts either a pip ``--platform`` tag (e.g. ``manylinux_2_17_x86_64``)
  or a Rust-like target triple (e.g. ``x86_64-unknown-linux-gnu``).
- It produces the exact set of pip knobs needed for cross-platform wheel
  downloads.
"""

from dataclasses import dataclass
import re
import sys
import sysconfig


class TargetResolutionError(ValueError):
    """Raised when a target spec cannot be resolved to pip-compatible tags."""


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """Build target configuration.

    :ivar platform_tag: pip ``--platform`` tag (PEP 425-style).
    :ivar python_version: Python version as ``MAJOR.MINOR``.
    :ivar implementation: pip implementation tag (e.g. ``cp``).
    :ivar abi: pip ABI tag (e.g. ``cp312``).
    """

    platform_tag: str
    python_version: str
    implementation: str
    abi: str


_PYVER_RE: re.Pattern[str] = re.compile(r"^(?P<maj>\d+)\.(?P<min>\d+)$")


def resolve_target_config(
    *,
    target: str,
    platform_tag_override: str | None,
    python_version_override: str | None,
    implementation_override: str | None,
    abi_override: str | None,
) -> TargetConfig:
    """Resolve user-supplied target arguments into a :class:`~TargetConfig`.

    :param target: A target triple or pip platform tag. ``native`` uses the host.
    :param platform_tag_override: Optional explicit pip platform tag override.
    :param python_version_override: Optional explicit Python version override.
    :param implementation_override: Optional explicit implementation override.
    :param abi_override: Optional explicit ABI override.
    :returns: Resolved target config.
    :raises TargetResolutionError: If the config cannot be resolved.
    """

    platform_tag: str
    if platform_tag_override is not None:
        platform_tag = _normalize_platform_tag(platform_tag_override)
    elif target == "native":
        platform_tag = _normalize_platform_tag(sysconfig.get_platform())
    else:
        platform_tag = _platform_tag_from_target_spec(target)

    python_version: str = _resolve_python_version(python_version_override)
    implementation: str = _resolve_implementation(implementation_override)
    abi: str = _resolve_abi(
        abi_override=abi_override,
        implementation=implementation,
        python_version=python_version,
    )

    return TargetConfig(
        platform_tag=platform_tag,
        python_version=python_version,
        implementation=implementation,
        abi=abi,
    )


def _resolve_python_version(python_version_override: str | None) -> str:
    """Resolve the target Python version.

    :param python_version_override: Optional explicit override.
    :returns: Version as ``MAJOR.MINOR``.
    :raises TargetResolutionError: If the override is invalid.
    """

    if python_version_override is None:
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    m = _PYVER_RE.match(python_version_override)
    if m is None:
        raise TargetResolutionError(
            f"Invalid --python-version {python_version_override!r}; expected 'MAJOR.MINOR'."
        )
    return f"{int(m.group('maj'))}.{int(m.group('min'))}"


def _resolve_implementation(implementation_override: str | None) -> str:
    """Resolve pip's implementation tag.

    :param implementation_override: Optional explicit override.
    :returns: Implementation tag (e.g. ``cp``).
    """

    if implementation_override is not None:
        return implementation_override

    impl_name: str = sys.implementation.name
    if impl_name == "cpython":
        return "cp"
    if impl_name == "pypy":
        return "pp"
    if len(impl_name) >= 2:
        return impl_name[0:2]
    return impl_name


def _resolve_abi(*, abi_override: str | None, implementation: str, python_version: str) -> str:
    """Resolve pip's ABI tag.

    :param abi_override: Optional explicit override.
    :param implementation: Implementation tag (e.g. ``cp``).
    :param python_version: Python version as ``MAJOR.MINOR``.
    :returns: ABI tag.
    :raises TargetResolutionError: If an ABI cannot be inferred.
    """

    if abi_override is not None:
        return abi_override

    if implementation != "cp":
        raise TargetResolutionError(
            "Non-CPython targets require an explicit --abi (and usually --implementation)."
        )

    m = _PYVER_RE.match(python_version)
    if m is None:
        raise TargetResolutionError(
            f"Internal error: python_version did not match MAJOR.MINOR: {python_version!r}"
        )
    maj: int = int(m.group("maj"))
    min_: int = int(m.group("min"))
    return f"cp{maj}{min_:02d}"


def _platform_tag_from_target_spec(target: str) -> str:
    """Convert a user-supplied target spec into pip's ``--platform`` tag.

    :param target: Target triple or platform tag.
    :returns: pip platform tag.
    :raises TargetResolutionError: If the target is not recognized.
    """

    normalized: str = _normalize_platform_tag(target)
    if _looks_like_pip_platform_tag(normalized) is True:
        return normalized

    parts: list[str] = target.split("-")
    if len(parts) < 3:
        raise TargetResolutionError(
            f"Unrecognized target spec {target!r}. Provide a Rust triple or pip --platform tag."
        )

    arch: str = parts[0]
    os_part: str = parts[2]
    env_part: str = parts[3] if len(parts) >= 4 else ""

    if os_part == "linux":
        return _linux_platform_tag(arch=arch, env_part=env_part)
    if os_part == "darwin":
        return _darwin_platform_tag(arch=arch)
    if os_part == "windows":
        return _windows_platform_tag(arch=arch)

    raise TargetResolutionError(
        f"Unrecognized OS in target triple {target!r} (os={os_part!r})."
    )


def _linux_platform_tag(*, arch: str, env_part: str) -> str:
    """Map a Rust-like Linux triple into a pip platform tag.

    :param arch: Rust arch component (e.g. ``x86_64``).
    :param env_part: Rust env component (e.g. ``gnu`` or ``musl``).
    :returns: pip platform tag.
    :raises TargetResolutionError: If the arch is not supported.
    """

    arch_map: dict[str, str] = {
        "x86_64": "x86_64",
        "aarch64": "aarch64",
        "armv7": "armv7l",
        "armv7l": "armv7l",
        "i686": "i686",
        "ppc64le": "ppc64le",
        "s390x": "s390x",
    }
    pip_arch: str | None = arch_map.get(arch)
    if pip_arch is None:
        raise TargetResolutionError(f"Unsupported Linux arch in target triple: {arch!r}")

    if env_part == "musl":
        return f"musllinux_1_2_{pip_arch}"
    return f"manylinux_2_17_{pip_arch}"


def _darwin_platform_tag(*, arch: str) -> str:
    """Map a Rust-like Darwin triple into a pip platform tag.

    :param arch: Rust arch component.
    :returns: pip platform tag.
    :raises TargetResolutionError: If the arch is not supported.
    """

    if arch == "x86_64":
        return "macosx_10_9_x86_64"
    if arch == "aarch64" or arch == "arm64":
        return "macosx_11_0_arm64"
    raise TargetResolutionError(f"Unsupported Darwin arch in target triple: {arch!r}")


def _windows_platform_tag(*, arch: str) -> str:
    """Map a Rust-like Windows triple into a pip platform tag.

    :param arch: Rust arch component.
    :returns: pip platform tag.
    :raises TargetResolutionError: If the arch is not supported.
    """

    if arch == "x86_64":
        return "win_amd64"
    if arch == "i686":
        return "win32"
    if arch == "aarch64" or arch == "arm64":
        return "win_arm64"
    raise TargetResolutionError(f"Unsupported Windows arch in target triple: {arch!r}")


def _looks_like_pip_platform_tag(platform_tag: str) -> bool:
    """Heuristically detect if a string looks like pip's ``--platform`` tag.

    :param platform_tag: Candidate platform tag.
    :returns: ``True`` if it looks like a pip platform tag.
    """

    if platform_tag.startswith("manylinux") is True:
        # Accept both PEP 600 (manylinux_2_17_x86_64) and legacy tags (manylinux2014_x86_64).
        return "_" in platform_tag
    if platform_tag.startswith("musllinux_") is True:
        return True
    if platform_tag.startswith("linux_") is True:
        return True
    if platform_tag.startswith("macosx_") is True:
        return True
    if platform_tag.startswith("win_") is True or platform_tag == "win32":
        return True
    return False


def _normalize_platform_tag(platform_tag: str) -> str:
    """Normalize common platform-tag spellings into pip's underscore form.

    :param platform_tag: Platform string (pip-style or sysconfig-style).
    :returns: Normalized platform tag.
    """

    # sysconfig uses e.g. "macosx-26.0-arm64" while pip expects "macosx_26_0_arm64".
    v: str = platform_tag.replace("-", "_").replace(".", "_")
    return v
