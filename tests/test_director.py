from pathlib import Path
import unittest

from video_ai.director import build_shot_plan
from video_ai.models import Transcript, Word


class DirectorTests(unittest.TestCase):
    def test_builds_fast_non_overlapping_scenes(self) -> None:
        tokens = [
            (0.00, 0.35, "Вчера"),
            (0.36, 0.72, "я"),
            (0.73, 1.15, "провалил"),
            (1.16, 1.55, "экзамен."),
            (1.70, 2.05, "Сначала"),
            (2.06, 2.38, "мне"),
            (2.39, 2.83, "казалось"),
            (2.84, 3.20, "что"),
            (3.21, 3.60, "всё"),
            (3.61, 4.10, "пропало,"),
            (4.20, 4.55, "но"),
            (4.56, 5.02, "потом"),
            (5.03, 5.48, "случилось"),
            (5.49, 5.90, "вот"),
            (5.91, 6.35, "что."),
        ]
        transcript = Transcript(words=[Word(*row) for row in tokens], language="ru")
        plan = build_shot_plan(transcript, Path("voice.mp3"), target_scene_seconds=2.0)

        self.assertGreaterEqual(len(plan.scenes), 2)
        self.assertEqual(plan.scenes[0].start, 0.0)
        self.assertAlmostEqual(plan.scenes[-1].end, 6.35, places=2)
        self.assertIn("вчера", plan.scenes[0].query)
        self.assertNotIn(" я ", f" {plan.scenes[0].query} ")

        for previous, current in zip(plan.scenes, plan.scenes[1:]):
            self.assertLessEqual(previous.end, current.start)


if __name__ == "__main__":
    unittest.main()
