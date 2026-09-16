from pathlib import Path
import tempfile
import unittest

from video_ai.director import build_shot_plan
from video_ai.io import load_shot_plan, save_shot_plan
from video_ai.models import Scene, ShotPlan, Transcript, Word


class V10EditingBrainTests(unittest.TestCase):
    def test_historical_context_prefers_archive_source(self) -> None:
        words = [
            Word(0.0, 0.4, "Хун"),
            Word(0.4, 0.8, "Сюцюань"),
            Word(0.8, 1.2, "начал"),
            Word(1.2, 1.7, "Тайпинское"),
            Word(1.7, 2.2, "восстание."),
        ]
        plan = build_shot_plan(
            Transcript(words=words, language="ru"),
            Path("voice.mp3"),
            target_scene_seconds=1.6,
            min_scene_seconds=0.8,
            max_scene_seconds=2.4,
        )
        self.assertTrue(plan.scenes)
        self.assertTrue(any(scene.source_mode == "historical_archive" for scene in plan.scenes))
        for scene in plan.scenes:
            if scene.source_mode == "historical_archive":
                self.assertNotEqual(scene.visual_mode, "video")

    def test_v10_fields_survive_save_load(self) -> None:
        scene = Scene(
            start=0.0,
            end=1.5,
            query="Qing dynasty examination",
            caption="Он провалил экзамен.",
            visual_mode="image",
            source_mode="historical_archive",
            motion_preset="dramatic_push",
            meme_filename=None,
            visual_description="19th century Qing scholar at imperial examination",
            search_queries=["Qing dynasty imperial examination"],
        )
        plan = ShotPlan(
            audio=Path("voice.mp3"),
            scenes=[scene],
            director_source="gemini",
            director_model="gemini-test",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            save_shot_plan(plan, path)
            loaded = load_shot_plan(path)
            self.assertEqual(loaded.scenes[0].source_mode, "historical_archive")
            self.assertEqual(loaded.scenes[0].motion_preset, "dramatic_push")
            self.assertEqual(loaded.director_source, "gemini")
            self.assertEqual(loaded.director_model, "gemini-test")

    def test_meme_scene_can_name_exact_local_file(self) -> None:
        scene = Scene(
            start=0.0,
            end=0.8,
            query="shocked disbelief reaction",
            caption="И тут он понял что произошло.",
            visual_mode="meme",
            source_mode="meme_library",
            motion_preset="none",
            meme_filename="shocked_disbelief_sunglasses_reaction.mp4",
        )
        self.assertEqual(scene.source_mode, "meme_library")
        self.assertEqual(scene.motion_preset, "none")
        self.assertTrue(scene.meme_filename.endswith(".mp4"))


if __name__ == "__main__":
    unittest.main()
