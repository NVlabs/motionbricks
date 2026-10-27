# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Metal generation and rig switching in the real OpenGL viewer."""

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp


@unittest.skipUnless(
    os.environ.get("MOTIONBRICKS_MPS_VIEWER_TESTS") == "1",
    "set MOTIONBRICKS_MPS_VIEWER_TESTS=1 to open the viewer",
)
class RealMetalViewerTests(unittest.TestCase):
    def test_real_demos(self):
        # Newton's GL viewer closes pyglet's process-wide event loop. Each app
        # normally runs in its own process, so exercise them the same way here.
        for case in ("_exercise_inbetweener", "_exercise_scene_preview"):
            with self.subTest(app=case):
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", f"tests.app.test_real_mps_viewer.RealMetalViewerTests.{case}", "-v"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=300,
                )
                self.assertEqual(result.returncode, 0, result.stdout)

    def _exercise_inbetweener(self):
        import newton.examples

        from app.inbetweener_demo import InbetweenerDemo
        from app.utils.cli import create_gl_parser
        from motionbricks.model.contract import ModelRig

        with wp.ScopedDevice("cpu"), \
             patch.dict(os.environ, {"MOTIONBRICKS_ONNX_PROVIDER": "mps"}), \
             patch.object(sys, "argv", ["motionbricks-viewer-test"]):
            viewer, _ = newton.examples.init(create_gl_parser())
            demo = None
            try:
                demo = InbetweenerDemo(viewer, ModelRig.UNITREE_G1)
                for rig in (ModelRig.UNITREE_G1, ModelRig.SOMA, ModelRig.UNITREE_G1):
                    if demo.rig is not rig:
                        demo.use_rig(rig)
                    self.assertEqual(demo.engine.provider, "mps")
                    demo.play()
                    demo.step(0.0)
                    self.assertTrue(demo.has_segment)
                    self.assertGreater(demo.output.num_frames, 0)
                    self.assertTrue(np.isfinite(demo.output.poses.numpy()[:demo.output.num_frames]).all())
                    demo.render()
                    demo.step(0.1)
                    demo.render()
                    self.assertGreater(demo.playback_time, 0.0)
            finally:
                if demo is not None and demo.engine is not None:
                    demo.engine.close()
                viewer.close()

    def _exercise_scene_preview(self):
        import newton.examples

        from app.scene_preview import ScenePreview
        from app.utils.cli import create_gl_parser
        from app.utils.input import InputFrame

        scene_dir = Path(__file__).resolve().parents[2] / "assets" / "scenes"
        with wp.ScopedDevice("cpu"), \
             patch.dict(os.environ, {"MOTIONBRICKS_ONNX_PROVIDER": "mps"}), \
             patch.object(sys, "argv", ["motionbricks-viewer-test"]):
            viewer, _ = newton.examples.init(create_gl_parser())
            scene = None
            try:
                scene = ScenePreview(viewer, scene_dir / "g1_gym.scene.json")
                for index, name in enumerate(("g1_gym.scene.json", "soma_gym.scene.json")):
                    if index:
                        scene.load_scene(scene_dir / name)
                    engine = scene._engine
                    self.assertEqual(engine.provider, "mps")
                    with patch.object(scene._input, "poll", return_value=InputFrame((0.0, 1.0), (0.0, 0.0), False)), \
                         patch.object(engine, "generate_animation", wraps=engine.generate_animation) as generate:
                        for _ in range(6):
                            scene.step()
                            scene.render()
                        self.assertGreater(generate.call_count, 0)
                    player = scene._scene_session.character.player
                    self.assertGreater(player.output.num_frames, 0)
                    self.assertTrue(np.isfinite(player.output.poses.numpy()[:player.output.num_frames]).all())
                    self.assertGreater(player.playback_time, 0.0)
            finally:
                if scene is not None:
                    scene.close()
                viewer.close()
