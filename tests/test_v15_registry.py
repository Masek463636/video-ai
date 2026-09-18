from video_ai.assets import MaterialRegistry


def test_material_registry_persists_and_releases_scene_usage() -> None:
    registry = MaterialRegistry()
    registry.register_scene(0, "https://cdn/a.mp4", "Phone reaction")
    registry.register_scene(1, "https://cdn/b.mp4", "Supermarket aisle")

    assert "https://cdn/a.mp4" in registry.used_urls()
    assert "https://cdn/b.mp4" in registry.used_urls()
    assert "phone reaction" in registry.used_titles()

    registry.release_scene(0)

    assert "https://cdn/a.mp4" not in registry.used_urls()
    assert "https://cdn/b.mp4" in registry.used_urls()
    assert "phone reaction" not in registry.used_titles()


def test_material_registry_tracks_same_source_used_by_two_scenes() -> None:
    registry = MaterialRegistry()
    registry.register_scene(0, "local:C:/memes/reaction.mp4", "reaction")
    registry.register_scene(1, "local:C:/memes/reaction.mp4", "reaction")

    registry.release_scene(1)

    assert "local:C:/memes/reaction.mp4" in registry.used_urls()
    registry.release_scene(0)
    assert "local:C:/memes/reaction.mp4" not in registry.used_urls()
