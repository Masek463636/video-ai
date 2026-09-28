"""Static camera for the fourth style; source footage keeps its own motion."""
from .models import ShotPlan

def apply_viral_motion(plan: ShotPlan) -> list[int]:
    changed = []
    for index, scene in enumerate(plan.scenes):
        if scene.motion_preset != "none" or scene.motion != "none":
            changed.append(index)
        scene.motion_preset = "none"
        scene.motion = "none"
    return changed
