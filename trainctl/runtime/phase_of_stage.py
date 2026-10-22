"""Maps a Lightning running stage to a run phase."""


def phase_of_stage(stage: str | None) -> str:
    """Returns the run phase of `stage`: `train`, `validate`, `test`, `predict` or `idle`.

    A sanity check counts as `validate`. `None` and the other stages count as `idle`.
    """
    if stage is None:
        return "idle"
    if stage in ("sanity_check", "validate"):
        return "validate"
    if stage in ("train", "test", "predict"):
        return stage
    return "idle"
