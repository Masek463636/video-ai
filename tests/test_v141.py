from pathlib import Path

from video_ai.assets import AssetCandidate, _historical_metadata_conflict
from video_ai.models import Scene, ShotPlan


def test_hard_lock_rejects_wrong_named_war() -> None:
    scene = Scene(
        start=0.0,
        end=2.0,
        query="Taiping Rebellion",
        caption="Тайпинское восстание в Китае",
        semantic_lock=True,
        required_entities=["Taiping Rebellion"],
        required_context=["China", "19th century"],
        semantic_fallback="Taiping Rebellion China 19th century archival illustration",
    )
    candidate = AssetCandidate(
        title="World War I soldiers in France",
        page_url="",
        download_url="https://example.com/a.jpg",
        mime="image/jpeg",
        width=1200,
        height=800,
        size=1000,
        kind="image",
        description="First World War trench scene, France, 1916",
        source="commons",
    )
    assert _historical_metadata_conflict(scene, candidate) is not None


def test_hard_lock_allows_matching_event_metadata() -> None:
    scene = Scene(
        start=0.0,
        end=2.0,
        query="Taiping Rebellion",
        caption="Тайпинское восстание в Китае",
        semantic_lock=True,
        required_entities=["Taiping Rebellion"],
        required_context=["China", "19th century"],
        semantic_fallback="Taiping Rebellion China 19th century archival illustration",
    )
    candidate = AssetCandidate(
        title="Taiping Rebellion battle in China",
        page_url="",
        download_url="https://example.com/a.jpg",
        mime="image/jpeg",
        width=1200,
        height=800,
        size=1000,
        kind="image",
        description="19th century Chinese illustration",
        source="commons",
    )
    assert _historical_metadata_conflict(scene, candidate) is None
