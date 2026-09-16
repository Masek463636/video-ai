from pathlib import Path
import tempfile
import unittest

from video_ai.audio_plan import build_audio_plan
from video_ai.models import Scene, ShotPlan
from video_ai.qc import failed_scene_indexes, inspect_plan


class V04Tests(unittest.TestCase):
    def test_audio_plan_adds_transition_and_impact(self) -> None:
        plan = ShotPlan(
            audio=Path("voice.mp3"),
            scenes=[
                Scene(0.0, 2.0, "student exam", caption="Я провалил экзамен", motion="zoom_in"),
                Scene(2.0, 4.0, "surprise", caption="Но вдруг всё изменилось", motion="pan_right"),
            ],
        )
        audio = build_audio_plan(plan)
        names = [cue.name for cue in audio.cues]
        self.assertIn("whoosh", names)
        self.assertIn("impact", names)
        self.assertEqual(audio.mood, "sad")

    def test_qc_flags_missing_assets(self) -> None:
        plan = ShotPlan(audio=Path("voice.mp3"), scenes=[Scene(0.0, 2.0, "x", caption="hello")])
        results = inspect_plan(plan)
        self.assertFalse(results[0].ok)
        self.assertEqual(failed_scene_indexes(results), {0})
        self.assertTrue(any("missing visual asset" in reason for reason in results[0].reasons))

    def test_qc_accepts_materialized_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "image.jpg"
            asset.write_bytes(b"not-a-real-jpeg-but-path-exists")
            plan = ShotPlan(
                audio=Path("voice.mp3"),
                scenes=[
                    Scene(
                        0.0, 2.0, "x", asset=str(asset), asset_kind="image",
                        caption="hello", focus_x=0.5, focus_y=0.5, focus_source="center",
                        asset_score=5.0, semantic_score=0.30,
                    )
                ],
            )
            results = inspect_plan(plan)
            self.assertTrue(results[0].ok)

    def test_qc_penalizes_very_weak_semantic_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "image.jpg"
            asset.write_bytes(b"x")
            plan = ShotPlan(
                audio=Path("voice.mp3"),
                scenes=[
                    Scene(
                        0.0, 2.0, "student exam", asset=str(asset), asset_kind="image",
                        caption="student taking an exam", focus_x=0.5, focus_y=0.5,
                        focus_source="center", asset_score=0.1, semantic_score=0.05,
                    )
                ],
            )
            result = inspect_plan(plan)[0]
            self.assertTrue(any("semantic match" in reason for reason in result.reasons))
            self.assertLess(result.score, 0.7)


if __name__ == "__main__":
    unittest.main()
