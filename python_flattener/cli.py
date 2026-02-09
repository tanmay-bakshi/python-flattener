"""Command line interface for python-flattener."""

import argparse
import pathlib

from python_flattener.builder import build_single_file
from python_flattener.target import TargetConfig, resolve_target_config


def main(argv: list[str] | None = None) -> int:
    """Run the python-flattener CLI.

    :param argv: Optional argv list (excluding program name).
    :returns: Exit code.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="python-flattener",
        description=(
            "Bundle a Python app + its dependencies into one self-extracting .py file."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser(
        "build",
        help="Build a single-file .py bundle.",
    )
    p_build.add_argument(
        "input",
        type=pathlib.Path,
        help="Path to a Python file or a module/package folder.",
    )
    p_build.add_argument(
        "-r",
        "--requirements",
        type=pathlib.Path,
        required=True,
        help="Path to requirements.txt (can be empty).",
    )
    p_build.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output path for the bundled single-file .py.",
    )
    p_build.add_argument(
        "--target",
        type=str,
        default="native",
        help=(
            "Target triple (e.g. x86_64-unknown-linux-gnu) or pip --platform tag "
            "(e.g. manylinux_2_17_x86_64). Use 'native' for the current host."
        ),
    )
    p_build.add_argument(
        "--platform-tag",
        type=str,
        default=None,
        help="Override pip --platform tag directly (advanced).",
    )
    p_build.add_argument(
        "--python-version",
        type=str,
        default=None,
        help="Target Python version as 'MAJOR.MINOR' (defaults to current).",
    )
    p_build.add_argument(
        "--implementation",
        type=str,
        default=None,
        help="Target implementation tag (e.g. cp, pp). Defaults to current.",
    )
    p_build.add_argument(
        "--abi",
        type=str,
        default=None,
        help="Target ABI tag (e.g. cp312). Defaults to current.",
    )
    p_build.add_argument(
        "--entry",
        type=str,
        default=None,
        help=(
            "Entry file path relative to the input directory (directory input only). "
            "If omitted, uses __main__.py or main.py when present."
        ),
    )
    p_build.add_argument(
        "--module",
        type=str,
        default=None,
        help=(
            "Entry module name to run (equivalent to python -m <module>). "
            "Useful for packages with relative imports."
        ),
    )

    ns = parser.parse_args(argv)
    if ns.command == "build":
        target_cfg: TargetConfig = resolve_target_config(
            target=ns.target,
            platform_tag_override=ns.platform_tag,
            python_version_override=ns.python_version,
            implementation_override=ns.implementation,
            abi_override=ns.abi,
        )

        build_single_file(
            input_path=ns.input,
            requirements_path=ns.requirements,
            output_path=ns.output,
            target=target_cfg,
            entry_relpath=ns.entry,
            entry_module=ns.module,
        )
        return 0

    raise AssertionError(f"Unhandled command: {ns.command}")
