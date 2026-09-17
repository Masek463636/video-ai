from pathlib import Path
import unittest

from video_ai.editing_grammar import apply_pre_asset_grammar, diversity_repair_indexes
from video_ai.models import Scene, ShotPlan
from video_ai.renderer import _visual_spans


class EditingGrammarV13Tests(unittest.TestCase):
    def test_visual_sequence_uses_action_before_literal_religious_noun(self) -> None:
        plan = ShotPlan(
            audio=Path("voice.mp3"),
            scenes=[
                Scene(0.0, 1.8, "wake", caption="Он проснулся после странного сна", visual_description="Jesus portrait"),
                Scene(1.8, 3.6, "belief", caption="и был уверен что он младший брат Иисуса", visual_description="Jesus portrait"),
                Scene(3.6, 5.4, "army", caption="а потом собрал крестьянскую армию", visual_description="religious followers"),
            ],
        )

        rewritten = apply_pre_asset_grammar(plan)

        self.assertEqual(rewritten, [0, 1, 2])
        self.assertIn("waking", plan.scenes[0].query.lower())
        self.assertIn("religious revelation", plan.scenes[1].visual_description.lower())
        self.assertIn("army", plan.scenes[2].query.lower())
        self.assertNotIn("portrait", plan.scenes[1].visual_description.lower())

    def test_diversity_marks_late_repeat_not_first_shot(self) -> None:
        same = str(Path("same.jpg").resolve())
        plan = ShotPlan(
            audio=Path("voice.mp3"),
            scenes=[
                Scene(0.0, 1.5, "a", asset=same, asset_kind="image"),
                Scene(1.5, 3.0, "b", asset=same, asset_kind="image"),
                Scene(3.0, 4.5, "c", asset=same, asset_kind="image"),
            ],
        )

        repairs = diversity_repair_indexes(plan, max_consecutive_seconds=2.6)

        self.assertNotIn(0, repairs)
        self.assertIn(1, repairs)
        self.assertIn(2, repairs)

    def test_renderer_merges_adjacent_identical_assets_into_one_span(self) -> None:
        same = str(Path("same.jpg").resolve())
        other = str(Path("other.jpg").resolve())
        plan = ShotPlan(
            audio=Path("voice.mp3"),
            scenes=[
                Scene(0.0, 1.5, "a", asset=same, asset_kind="image", caption="one"),
                Scene(1.5, 3.0, "b", asset=same, asset_kind="image", caption="two"),
                Scene(3.0, 4.5, "c", asset=other, asset_kind="image", caption="three"),
            ],
        )

        spans = _visual_spans(plan, 4.5)

        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0][1], 0.0)
        self.assertAlmostEqual(spans[0][2], 3.0)
        self.assertAlmostEqual(spans[1][1], 3.0)
        self.assertAlmostEqual(spans[1][2], 4.5)


if __name__ == "__main__":
    unittest.main()
