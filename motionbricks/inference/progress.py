# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Track TensorRT build progress safely across threads."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

try:
    import tensorrt as trt
except ImportError:
    trt = None


@dataclass(frozen=True, slots=True)
class TensorRTBuildProgress:
    """Progress through the TensorRT optimizer's current phase."""

    phase_name: str | None = None
    completed_steps: int = 0
    total_steps: int = 0

    @property
    def fraction(self) -> float:
        """Completed share of the phase, from 0 to 1."""
        if self.total_steps == 0:
            return 0.0
        return min(1.0, self.completed_steps / self.total_steps)


_MonitorBase = trt.IProgressMonitor if trt is not None else object


class TensorRTBuildProgressMonitor(_MonitorBase):
    """Collect TensorRT optimizer callbacks for polling from another thread."""

    def __init__(self) -> None:
        if trt is not None:
            super().__init__()
        self._lock = Lock()
        self._phases: dict[str, tuple[int, int]] = {}
        self._active_phase: str | None = None
        self._cancel_requested = False

    def phase_start(
        self,
        phase_name: str,
        _parent_phase: str | None,
        num_steps: int,
    ) -> None:
        """Record the start of an optimizer phase. TensorRT calls it.

        Args:
            phase_name: Name of the phase.
            _parent_phase: Name of the enclosing phase, or None. It is ignored.
            num_steps: Number of steps in the phase.
        """
        with self._lock:
            self._phases[phase_name] = (0, num_steps)
            self._active_phase = phase_name

    def step_complete(self, phase_name: str, step: int) -> bool:
        """Record a finished step, and return False once a cancel is requested.

        TensorRT calls it. A False return stops the build.

        Args:
            phase_name: Name of the phase.
            step: Index of the finished step, from zero.
        """
        with self._lock:
            phase = self._phases.get(phase_name)
            if phase is not None:
                _, total_steps = phase
                completed_steps = min(step + 1, total_steps)
                self._phases[phase_name] = (completed_steps, total_steps)
                self._active_phase = phase_name
            return not self._cancel_requested

    def phase_finish(self, phase_name: str) -> None:
        """Mark every step of the phase as done. TensorRT calls it.

        Args:
            phase_name: Name of the phase.
        """
        with self._lock:
            phase = self._phases.get(phase_name)
            if phase is None:
                return
            _, total_steps = phase
            self._phases[phase_name] = (total_steps, total_steps)
            self._active_phase = phase_name

    def snapshot(self) -> TensorRTBuildProgress:
        """Return the progress of the unfinished phase with the most steps.

        When every phase is done, it returns the last active phase.
        """
        with self._lock:
            if self._active_phase is None:
                return TensorRTBuildProgress()

            active_phases = [
                (name, completed_steps, total_steps)
                for name, (completed_steps, total_steps) in self._phases.items()
                if completed_steps < total_steps
            ]
            if active_phases:
                phase_name, completed_steps, total_steps = max(
                    active_phases,
                    key=lambda phase: phase[2],
                )
            else:
                phase_name = self._active_phase
                completed_steps, total_steps = self._phases[phase_name]
            return TensorRTBuildProgress(
                phase_name,
                completed_steps,
                total_steps,
            )

    def request_cancel(self) -> None:
        """Ask TensorRT to stop the build after its current step."""
        with self._lock:
            self._cancel_requested = True
