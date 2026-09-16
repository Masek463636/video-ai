from pathlib import Path
import tempfile
import unittest

from video_ai.director import build_shot_plan
from video_ai.meme_library import search_memes
from video_ai.models import Transcript, Word


class V08Tests(unittest.TestCase):
    def test_director_emits_structured_visual_intent(self) -> None:
        words = [
            Word(0.0, 0.4, "В"), Word(0.4, 0.8, "Китае"), Word(0.8, 1.2, "он"),
            Word(1.2, 1.7, "провалил"), Word(1.7, 2.2, "экзамен."),
            Word(2.3, 2.7, "А"), Word(2.7, 3.1, "потом"), Word(3.1, 3.5, "вдруг"),
            Word(3.5, 4.0, "увидел"), Word(4.0, 4.5, "видение."),
        ]
        plan = build_shot_plan(Transcript(words=words, language="ru"), Path("voice.mp3"), target_scene_seconds=1.8)
        self.assertGreaterEqual(len(plan.scenes), 2)
        for scene in plan.scenes:
            self.assertTrue(scene.visual_description)
            self.assertGreaterEqual(len(scene.search_queries), 2)
            self.assertIn(scene.visual_mode, {"image", "video", "meme"})
        self.assertTrue(any("China" in q for q in plan.scenes[0].search_queries))

    def test_meme_library_prefers_matching_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "surprised_reaction.mp4").write_bytes(b"x" * 2048)
            (root / "sad_face.mp4").write_bytes(b"x" * 2048)
            results = search_memes(root, "surprised reaction", limit=2)
            self.assertEqual(results[0].path.name, "surprised_reaction.mp4")
            self.assertEqual(results[0].kind, "video")


if __name__ == "__main__":
    unittest.main()
