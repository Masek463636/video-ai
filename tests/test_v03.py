from pathlib import Path
import tempfile
import unittest

from video_ai.assets import AssetCandidate, _repeat_penalty, _score
from video_ai.io import load_shot_plan, save_shot_plan
from video_ai.models import Scene, ShotPlan


class V03Tests(unittest.TestCase):
    def test_relevant_candidate_beats_irrelevant_one(self) -> None:
        relevant = AssetCandidate(
            title="Student taking exam in classroom",
            page_url="",
            download_url="https://example.com/a.jpg",
            mime="image/jpeg",
            width=1400,
            height=2100,
            size=1000,
            kind="image",
            description="Young student writing university exam at desk",
            license="CC BY",
        )
        irrelevant = AssetCandidate(
            title="Mountain landscape",
            page_url="",
            download_url="https://example.com/b.jpg",
            mime="image/jpeg",
            width=3000,
            height=2000,
            size=1000,
            kind="image",
            description="Snowy mountain and lake",
            license="CC BY",
        )
        query = "student exam classroom"
        caption = "Я немного расстроился из-за экзамена"
        self.assertGreater(_score(relevant, query, caption), _score(irrelevant, query, caption))

    def test_repeat_penalty_detects_similar_titles(self) -> None:
        penalty = _repeat_penalty(
            "Student taking exam in classroom",
            ["Student writing exam in classroom"],
        )
        self.assertGreater(penalty, 0.0)

    def test_focus_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = ShotPlan(
                audio=root / "voice.mp3",
                scenes=[
                    Scene(
                        start=0.0,
                        end=2.0,
                        query="student exam",
                        caption="Экзамен",
                        focus_x=0.21,
                        focus_y=0.43,
                        focus_source="face",
                    )
                ],
            )
            path = root / "plan.json"
            save_shot_plan(plan, path)
            loaded = load_shot_plan(path)
            self.assertAlmostEqual(loaded.scenes[0].focus_x or 0, 0.21)
            self.assertAlmostEqual(loaded.scenes[0].focus_y or 0, 0.43)
            self.assertEqual(loaded.scenes[0].focus_source, "face")


if __name__ == "__main__":
    unittest.main()
