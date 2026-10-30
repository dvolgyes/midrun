"""Behavioral tests for the RuntimeSnapshotChanges keys of StateStore.update."""

from dataclasses import fields

from midrun.runtime.state.runtime_snapshot import RuntimeSnapshot
from midrun.runtime.state.runtime_snapshot_changes import RuntimeSnapshotChanges


def test_changes_cover_every_snapshot_field_that_update_does_not_set() -> None:
    snapshot_fields = {field.name for field in fields(RuntimeSnapshot)}
    assert set(RuntimeSnapshotChanges.__annotations__) == snapshot_fields - {
        "generation",
        "updated_at",
    }
