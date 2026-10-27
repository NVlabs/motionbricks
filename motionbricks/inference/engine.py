# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the native inference backend for the current platform."""

import sys

if sys.platform == "darwin":
    from .onnx_engine import InferenceEngine
else:
    from .tensorrt_engine import InferenceEngine

__all__ = ["InferenceEngine"]
