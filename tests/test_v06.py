from pathlib import Path
import unittest

from video_ai.director import build_shot_plan
from video_ai.models import Transcript, Word


class V06Tests(unittest.TestCase):
    def test_global_historical_context_is_carried_into_every_scene(self) -> None:
        words = [
            Word(0.00, 0.40, "В"),
            Word(0.41, 0.90, "Китае"),
            Word(0.91, 1.50, "XIX"),
            Word(1.51, 2.05, "века"),
            Word(2.06, 2.55, "жил"),
            Word(2.56, 3.00, "Хун"),
            Word(3.01, 3.45, "Сюцюань."),
            Word(3.60, 4.00, "Он"),
            Word(4.01, 4.60, "провалил"),
            Word(4.61, 5.30, "госэкзамен"),
            Word(5.31, 5.90, "и"),
            Word(5.91, 6.60, "сильно"),
            Word(6.61, 7.30, "расстроился."),
        ]
        transcript = Transcript(words=words, language="ru")
        plan = build_shot_plan(transcript, Path("voice.mp3"), target_scene_seconds=2.1)

        self.assertGreaterEqual(len(plan.scenes), 2)
        for scene in plan.scenes:
            query = scene.query.lower()
            self.assertIn("china", query)
            self.assertIn("19th century", query)
            self.assertIn("historical archival illustration", query)

        exam_query = plan.scenes[-1].query.lower()
        self.assertIn("civil service examination", exam_query)

    def test_modern_story_does_not_force_historical_style(self) -> None:
        words = [
            Word(0.0, 0.5, "Парень"),
            Word(0.51, 1.0, "смотрит"),
            Word(1.01, 1.6, "сообщение"),
            Word(1.61, 2.2, "на"),
            Word(2.21, 2.8, "телефоне."),
        ]
        plan = build_shot_plan(Transcript(words=words, language="ru"), Path("voice.mp3"))
        query = plan.scenes[0].query.lower()
        self.assertIn("smartphone", query)
        self.assertNotIn("historical archival illustration", query)


if __name__ == "__main__":
    unittest.main()
