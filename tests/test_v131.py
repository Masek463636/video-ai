from pathlib import Path
import unittest

from video_ai.material_brain import (
    diversity_summary,
    find_duplicate_scenes,
    hash_similarity,
    prepare_diversity_repair,
)
from video_ai.models import Scene, ShotPlan


class MaterialBrainV131Tests(unittest.TestCase):
    def test_exact_asset_reuse_is_repaired_after_first_use(self) -> None:
        shared = str(Path("shared.jpg").resolve())
        plan = ShotPlan(
            audio=Path("voice.mp3"),
            scenes=[
                Scene(0.0, 1.5, "a", asset=shared, asset_kind="image"),
                Scene(1.5, 3.0, "b", asset=shared, asset_kind="image"),
                Scene(3.0, 4.5, "c", asset=shared, asset_kind="image"),
            ],
        )

        repairs, matches = find_duplicate_scenes(plan)

        self.assertEqual(repairs, [1, 2])
        self.assertEqual([m.reason for m in matches], ["same_asset", "same_asset"])
        self.assertTrue(all(m.original_scene == 0 for m in matches))

    def test_hash_similarity_detects_near_identical_visuals(self) -> None:
        all_ones = (1 << 256) - 1
        one_bit_changed = all_ones ^ 1
        many_bits_changed = all_ones ^ ((1 << 80) - 1)

        self.assertGreater(hash_similarity(all_ones, one_bit_changed), 0.99)
        self.assertLess(hash_similarity(all_ones, many_bits_changed), 0.70)

    def test_repair_changes_search_framing_for_generic_scene(self) -> None:
        plan = ShotPlan(
            audio=Path("voice.mp3"),
            scenes=[
                Scene(
                    0.0,
                    1.5,
                    "stress",
                    caption="Он был в стрессе после провала",
                    query="stressed man",
                    search_queries=["stressed man"],
                    visual_description="stressed man reaction",
                    visual_mode="image",
                    source_mode="generic_image",
                )
            ],
        )

        prepare_diversity_repair(plan, [0], pass_index=1)
        scene = plan.scenes[0]

        self.assertNotEqual(scene.query, "stressed man")
        self.assertIn("distinct", scene.visual_description.lower())
        self.assertEqual(scene.visual_mode, "video")
        self.assertEqual(scene.source_mode, "stock_video")

    def test_summary_counts_exact_path_duplicates(self) -> None:
        shared = str(Path("same.png").resolve())
        other = str(Path("other.png").resolve())
        plan = ShotPlan(
            audio=Path("voice.mp3"),
            scenes=[
                Scene(0.0, 1.0, "a", asset=shared, asset_kind="image"),
                Scene(1.0, 2.0, "b", asset=shared, asset_kind="image"),
                Scene(2.0, 3.0, "c", asset=other, asset_kind="image"),
            ],
        )

        summary = diversity_summary(plan)

        self.assertEqual(summary["materialized_scenes"], 3)
        self.assertEqual(summary["unique_visuals"], 2)
        self.assertEqual(summary["duplicate_scenes"], [1])


if __name__ == "__main__":
    unittest.main()
