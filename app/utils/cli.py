# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared command line and logging setup for the MotionBricks apps."""

from __future__ import annotations

import argparse
import logging

import warp as wp


_HIDDEN_NEWTON_OPTIONS = frozenset(
    {
        "--viewer",
        "--rerun-address",
        "--output-path",
        "--num-frames",
        "--render-fps",
        "--headless",
        "--test",
        "--paused",
        "--benchmark",
        "--warp-config",
        "--realtime",
    }
)
# Loggers that follow --quiet and --verbose. Other libraries only show warnings.
_APP_LOGGERS = ("__main__", "app", "motionbricks")


class _MessageFormatter(logging.Formatter):
    """Show messages as they are, with the level in front of warnings and errors."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return f"{record.levelname}: {message}" if record.levelno >= logging.WARNING else message


def create_gl_parser() -> argparse.ArgumentParser:
    """Use the OpenGL viewer, hide Newton's extra options, and add output levels."""

    import newton.examples

    parser = newton.examples.create_parser()
    for action in parser._actions:
        if _HIDDEN_NEWTON_OPTIONS.intersection(action.option_strings):
            action.help = argparse.SUPPRESS
        if "--viewer" in action.option_strings:
            action.choices = ("gl",)
        if "--quiet" in action.option_strings:
            action.help = "Show only warnings and errors."
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed messages, such as each clip load, inference output, and Warp module loads.",
    )
    return parser


def add_model_precision_argument(parser: argparse.ArgumentParser) -> None:
    """Add --model-precision, which selects the bundled FP32 or mixed FP16 models."""

    parser.add_argument(
        "--model-precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Precision of the bundled ONNX models.",
    )


def configure_logging(args: argparse.Namespace) -> None:
    """Show app messages, only problems with --quiet, or every detail with --verbose.

    Warp's module load messages are details too, so they only show with --verbose.
    """

    handler = logging.StreamHandler()
    handler.setFormatter(_MessageFormatter())
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(level)
    if not args.verbose:
        wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)
