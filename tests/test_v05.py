from pathlib import Path
import tempfile
import unittest

from video_ai.assets import ensure_visual_coverage
from video_ai.models import Scene, ShotPlan


class V05Tests(unittest.TestCase):
    def test_early_blank_scenes_reuse_nearest_future_visual(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "scene_003.jpg"
            asset.write_bytes(b"x" * 2048)
            plan = ShotPlan(
                audio=Path("voice.mp3"),
                scenes=[
                    Scene(0.0, 2.0, "a", caption="one"),
                    Scene(2.0, 4.0, "b", caption="two"),
                    Scene(4.0, 6.0, "c", caption="three"),
                    Scene(
                        6.0,
                        8.0,
                        "d",
                        asset=str(asset),
                        asset_kind="image",
                        caption="four",
                        focus_x=0.5,
                        focus_y=0.5,
                        focus_source="center",
                    ),
                ],
            )

            filled = ensure_visual_coverage(plan)

            self.assertEqual(filled, {0, 1, 2})
            for index in (0, 1, 2):
                self.assertEqual(plan.scenes[index].asset, str(asset))
                self.assertEqual(plan.scenes[index].asset_kind, "image")
                self.assertTrue((plan.scenes[index].focus_source or "").startswith("fallback_nearest:"))

    def test_existing_visual_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jpg"
            second = Path(directory) / "second.jpg"
            first.write_bytes(b"a" * 2048)
            second.write_bytes(b"b" * 2048)
            plan = ShotPlan(
                audio=Path("voice.mp3"),
                scenes=[
                    Scene(0.0, 2.0, "a", asset=str(first), asset_kind="image"),
                    Scene(2.0, 4.0, "b", asset=str(second), asset_kind="image"),
                ],
            )

            filled = ensure_visual_coverage(plan)

            self.assertEqual(filled, set())
            self.assertEqual(plan.scenes[0].asset, str(first))
            self.assertEqual(plan.scenes[1].asset, str(second))


if __name__ == "__main__":
    unittest.main()
