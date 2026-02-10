"""Command line interface for python-flattener."""

import argparse
import logging
import pathlib
import sys

from python_flattener.builder import build_single_file
from python_flattener.target import TargetConfig, resolve_target_config


def _configure_logging(*, verbose: int, quiet: int) -> logging.Logger:
    """Configure the python-flattener logger.

    :param verbose: Verbosity count (0+).
    :param quiet: Quietness count (0+).
    :returns: Configured logger.
    """

    level: int = logging.INFO
    if quiet >= 2:
        level = logging.ERROR
    elif quiet >= 1:
        level = logging.WARNING
    elif verbose >= 1:
        level = logging.DEBUG

    logger: logging.Logger = logging.getLogger("python_flattener")
    logger.setLevel(level)
    logger.propagate = False

    handler: logging.Handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


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
    p_build.add_argument(
        "--in-memory-runtime",
        action="store_true",
        help=(
            "Prefer an in-memory runtime that avoids filesystem extraction (Linux only). "
            "This is useful for containers with read-only filesystems."
        ),
    )
    p_build.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose logging. Pass multiple times for more detail.",
    )
    p_build.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Reduce logging. Pass multiple times to suppress more output.",
    )

    ns = parser.parse_args(argv)
    if ns.command == "build":
        logger: logging.Logger = _configure_logging(verbose=ns.verbose, quiet=ns.quiet)
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
            logger=logger,
            prefer_in_memory_runtime=ns.in_memory_runtime,
        )
        return 0

    raise AssertionError(f"Unhandled command: {ns.command}")
