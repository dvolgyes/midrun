"""Behavioral tests for midrun.hooks.manager's baseline bootstrap (T1)."""

import errno
import json
import os
import shutil
import stat
import threading
from pathlib import Path

import pytest
from loguru import logger

from midrun.hooks.catalog import CALLBACKS, CATALOGUE_VERSION, MODALITIES
from midrun.hooks.manager.bootstrap_hooks_tree import bootstrap_hooks_tree
from midrun.hooks.manager.hooks_bootstrap_error import HooksBootstrapError
import midrun.hooks.manager.publish_fresh_tree as publish_fresh_tree_module
from midrun.hooks.manager.publish_fresh_tree import publish_fresh_tree

_SLOT_COUNT = len(CALLBACKS) * len(MODALITIES)


@pytest.fixture()
def log():
    return logger


def _all_slot_paths(live_dir) -> list:
    return [
        live_dir / modality / f"{spec.name}.sh"
        for spec in CALLBACKS
        for modality in MODALITIES
    ]


def test_no_source_creates_74_disabled_scripts_and_readme(tmp_path, log) -> None:
    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(base_dir, None, session_id="s-1", log=log).live_dir

    assert live_dir == base_dir / "hooks"
    slots = _all_slot_paths(live_dir)
    assert len(slots) == _SLOT_COUNT == 74
    for script in slots:
        assert script.is_file()
        assert stat.S_IMODE(script.stat().st_mode) & 0o111 == 0, (
            f"{script} must not be executable"
        )
        assert script.read_text().startswith("#!/usr/bin/env bash\n")
    assert (live_dir / "README.md").is_file()
    marker = json.loads((base_dir / "midrun-hooks-state.json").read_text())
    assert marker["session_id"] == "s-1"
    assert marker["source_dir"] is None
    provenance = json.loads((live_dir / ".midrun-provenance.json").read_text())
    assert provenance["session_id"] == "s-1"


def test_partial_source_fills_only_missing_slots_and_preserves_bytes_and_mode(
    tmp_path, log
) -> None:
    source = tmp_path / "baseline"
    (source / "light").mkdir(parents=True)
    (source / "heavy").mkdir(parents=True)
    custom_light = source / "light" / "on_fit_start.sh"
    custom_light.write_text("#!/usr/bin/env bash\necho custom-light\n")
    custom_light.chmod(0o755)
    custom_heavy = source / "heavy" / "on_train_batch_end.sh"
    custom_heavy.write_text("#!/usr/bin/env bash\necho custom-heavy\n")
    custom_heavy.chmod(0o750)

    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(
        base_dir, source, session_id="s-2", log=log
    ).live_dir

    live_light = live_dir / "light" / "on_fit_start.sh"
    assert live_light.read_bytes() == custom_light.read_bytes()
    assert stat.S_IMODE(live_light.stat().st_mode) == 0o755

    live_heavy = live_dir / "heavy" / "on_train_batch_end.sh"
    assert live_heavy.read_bytes() == custom_heavy.read_bytes()
    assert stat.S_IMODE(live_heavy.stat().st_mode) == 0o750

    for script in _all_slot_paths(live_dir):
        if script in (live_light, live_heavy):
            continue
        assert stat.S_IMODE(script.stat().st_mode) & 0o111 == 0, (
            f"{script} must stay disabled"
        )


def test_helper_files_are_copied_recursively_with_preserved_mode(tmp_path, log) -> None:
    source = tmp_path / "baseline"
    (source / "helpers" / "nested").mkdir(parents=True)
    helper = source / "helpers" / "nested" / "notify.sh"
    helper.write_text("echo hi\n")
    helper.chmod(0o700)

    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(
        base_dir, source, session_id="s-3", log=log
    ).live_dir

    live_helper = live_dir / "helpers" / "nested" / "notify.sh"
    assert live_helper.read_text() == "echo hi\n"
    assert stat.S_IMODE(live_helper.stat().st_mode) == 0o700


def test_symlink_in_source_is_rejected_and_nothing_is_published(tmp_path, log) -> None:
    source = tmp_path / "baseline"
    source.mkdir()
    target = tmp_path / "outside.sh"
    target.write_text("echo x\n")
    (source / "link.sh").symlink_to(target)

    base_dir = tmp_path / "run"
    with pytest.raises(HooksBootstrapError, match="symlink"):
        bootstrap_hooks_tree(base_dir, source, session_id="s-4", log=log)

    assert not (base_dir / "hooks").exists()
    assert not (base_dir / "midrun-hooks-state.json").exists()
    assert list(base_dir.glob(".hooks-staging-*")) == []


def test_special_file_in_source_is_rejected(tmp_path, log) -> None:
    source = tmp_path / "baseline"
    source.mkdir()
    os.mkfifo(source / "pipe")

    base_dir = tmp_path / "run"
    with pytest.raises(HooksBootstrapError, match="non-regular file"):
        bootstrap_hooks_tree(base_dir, source, session_id="s-5", log=log)
    assert not (base_dir / "hooks").exists()


def test_setuid_bit_in_source_is_rejected(tmp_path, log) -> None:
    source = tmp_path / "baseline"
    source.mkdir()
    script = source / "weird.sh"
    script.write_text("echo x\n")
    script.chmod(0o755 | stat.S_ISUID)

    base_dir = tmp_path / "run"
    with pytest.raises(HooksBootstrapError, match="privileged mode bits"):
        bootstrap_hooks_tree(base_dir, source, session_id="s-6", log=log)
    assert not (base_dir / "hooks").exists()


def test_source_equal_to_run_directory_is_rejected(tmp_path, log) -> None:
    base_dir = tmp_path / "run"
    base_dir.mkdir()

    with pytest.raises(
        HooksBootstrapError, match="must not equal, contain, or be contained by"
    ):
        bootstrap_hooks_tree(base_dir, base_dir, session_id="s-7", log=log)


def test_source_inside_run_directory_is_rejected(tmp_path, log) -> None:
    base_dir = tmp_path / "run"
    nested_source = base_dir / "baseline"
    nested_source.mkdir(parents=True)

    with pytest.raises(
        HooksBootstrapError, match="must not equal, contain, or be contained by"
    ):
        bootstrap_hooks_tree(base_dir, nested_source, session_id="s-8", log=log)


def test_run_directory_inside_source_is_rejected(tmp_path, log) -> None:
    source = tmp_path / "baseline"
    base_dir = source / "nested" / "run"
    source.mkdir()

    with pytest.raises(
        HooksBootstrapError, match="must not equal, contain, or be contained by"
    ):
        bootstrap_hooks_tree(base_dir, source, session_id="s-9", log=log)


def test_repeat_setup_reuses_live_tree_and_never_restores_deleted_scripts(
    tmp_path, log
) -> None:
    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(base_dir, None, session_id="s-10", log=log).live_dir

    deleted = live_dir / "light" / "on_fit_start.sh"
    deleted.unlink()
    edited = live_dir / "light" / "on_fit_end.sh"
    edited.write_text("#!/usr/bin/env bash\necho edited\n")
    edited.chmod(0o755)

    live_dir_again = bootstrap_hooks_tree(
        base_dir, None, session_id="s-10-again", log=log
    ).live_dir

    assert live_dir_again == live_dir
    assert not deleted.exists()
    assert edited.read_text() == "#!/usr/bin/env bash\necho edited\n"
    assert stat.S_IMODE(edited.stat().st_mode) & 0o111 != 0
    marker = json.loads((base_dir / "midrun-hooks-state.json").read_text())
    assert marker["session_id"] == "s-10"  # first session's marker, not overwritten


def test_a_resumed_tree_keeps_the_first_session_id(tmp_path, log) -> None:
    base_dir = tmp_path / "run"
    first = bootstrap_hooks_tree(base_dir, None, session_id="s-first", log=log)
    resumed = bootstrap_hooks_tree(base_dir, None, session_id="s-again", log=log)

    assert first.session_id == "s-first"
    assert resumed.session_id == "s-first"


def test_baseline_edits_after_bootstrap_do_not_propagate_on_resume(
    tmp_path, log
) -> None:
    source = tmp_path / "baseline"
    (source / "light").mkdir(parents=True)
    (source / "heavy").mkdir(parents=True)
    (source / "light" / "on_fit_start.sh").write_text(
        "#!/usr/bin/env bash\necho original\n"
    )

    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(
        base_dir, source, session_id="s-11", log=log
    ).live_dir

    (source / "light" / "on_fit_start.sh").write_text(
        "#!/usr/bin/env bash\necho changed-after\n"
    )

    live_dir_again = bootstrap_hooks_tree(
        base_dir, source, session_id="s-11-again", log=log
    ).live_dir

    assert live_dir_again == live_dir
    assert (
        live_dir / "light" / "on_fit_start.sh"
    ).read_text() == "#!/usr/bin/env bash\necho original\n"


def test_unknown_existing_live_tree_is_a_conflict_and_is_left_untouched(
    tmp_path, log
) -> None:
    base_dir = tmp_path / "run"
    live_dir = base_dir / "hooks"
    live_dir.mkdir(parents=True)
    (live_dir / "mystery.txt").write_text("not ours\n")

    with pytest.raises(HooksBootstrapError, match="not a Midrun-owned hooks tree"):
        bootstrap_hooks_tree(base_dir, None, session_id="s-12", log=log)

    assert (live_dir / "mystery.txt").read_text() == "not ours\n"
    assert not (base_dir / "midrun-hooks-state.json").exists()


def test_interrupted_publish_recovers_from_provenance_without_recopying(
    tmp_path, log
) -> None:
    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(base_dir, None, session_id="s-13", log=log).live_dir
    # Simulate a crash between the atomic rename and the external marker write.
    (base_dir / "midrun-hooks-state.json").unlink()
    sentinel = live_dir / "light" / "on_fit_start.sh"
    sentinel.write_text("#!/usr/bin/env bash\necho untouched-by-recovery\n")

    live_dir_again = bootstrap_hooks_tree(
        base_dir, None, session_id="s-13-recovered", log=log
    ).live_dir

    assert live_dir_again == live_dir
    marker = json.loads((base_dir / "midrun-hooks-state.json").read_text())
    assert (
        marker["session_id"] == "s-13"
    )  # recovered from the live tree's own provenance
    assert sentinel.read_text() == "#!/usr/bin/env bash\necho untouched-by-recovery\n"


def test_lost_publish_race_reports_unpublished_and_cleans_up_staging(
    tmp_path, log
) -> None:
    base_dir = tmp_path / "run"
    base_dir.mkdir()
    live_dir = base_dir / "hooks"
    live_dir.mkdir()
    (live_dir / "already-here.txt").write_text("winner\n")

    assert publish_fresh_tree(base_dir, live_dir, None, "s-14", log) is False

    assert list(base_dir.glob(".hooks-staging-*")) == []
    assert not (base_dir / "midrun-hooks-state.json").exists()
    assert (live_dir / "already-here.txt").read_text() == "winner\n"


def test_lost_publish_race_cleans_up_the_staging_copy_of_a_read_only_baseline(
    tmp_path, log
) -> None:
    source = tmp_path / "baseline"
    (source / "light").mkdir(parents=True)
    (source / "light" / "on_fit_start.sh").write_text("#!/usr/bin/env bash\n")
    (source / "light").chmod(0o555)
    source.chmod(0o555)
    base_dir = tmp_path / "run"
    base_dir.mkdir()
    live_dir = base_dir / "hooks"
    live_dir.mkdir()
    (live_dir / "already-here.txt").write_text("winner\n")

    assert publish_fresh_tree(base_dir, live_dir, source, "s-14", log) is False

    assert list(base_dir.glob(".hooks-staging-*")) == []


def test_losing_the_publish_race_resumes_the_winners_tree(
    tmp_path, monkeypatch, log
) -> None:
    base_dir = tmp_path / "run"
    real_write_readme = publish_fresh_tree_module.write_readme_if_absent

    def winner_publishes_first(staging_dir):
        monkeypatch.setattr(
            publish_fresh_tree_module, "write_readme_if_absent", real_write_readme
        )
        bootstrap_hooks_tree(base_dir, None, session_id="s-winner", log=log)
        real_write_readme(staging_dir)

    monkeypatch.setattr(
        publish_fresh_tree_module, "write_readme_if_absent", winner_publishes_first
    )

    loser = bootstrap_hooks_tree(base_dir, None, session_id="s-loser", log=log)

    assert loser.session_id == "s-winner"
    assert loser.live_dir == base_dir / "hooks"
    assert list(base_dir.glob(".hooks-staging-*")) == []
    marker = json.loads((base_dir / "midrun-hooks-state.json").read_text())
    assert marker["session_id"] == "s-winner"


def test_concurrent_bootstraps_all_resolve_to_one_tree(tmp_path, log) -> None:
    base_dir = tmp_path / "run"
    workers = 4
    barrier = threading.Barrier(workers)
    results: dict[int, object] = {}

    def bootstrap(rank: int) -> None:
        barrier.wait()
        try:
            results[rank] = bootstrap_hooks_tree(
                base_dir, None, session_id=f"s-rank-{rank}", log=log
            )
        except HooksBootstrapError as exc:
            results[rank] = exc

    threads = [threading.Thread(target=bootstrap, args=(r,)) for r in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    errors = [r for r in results.values() if isinstance(r, Exception)]
    assert errors == []
    assert len({tree.session_id for tree in results.values()}) == 1
    assert list(base_dir.glob(".hooks-staging-*")) == []


def test_source_file_changing_during_copy_is_rejected(
    tmp_path, monkeypatch, log
) -> None:
    source = tmp_path / "baseline"
    (source / "light").mkdir(parents=True)
    (source / "heavy").mkdir(parents=True)
    changing = source / "light" / "on_fit_start.sh"
    changing.write_text("#!/usr/bin/env bash\necho v1\n")

    real_copyfile = shutil.copyfile

    def flaky_copyfile(src, dst):
        real_copyfile(src, dst)
        if str(src) == str(changing):
            with open(src, "a") as fh:
                fh.write("echo v2\n")

    monkeypatch.setattr(
        "midrun.hooks.manager.copy_source_tree.shutil.copyfile", flaky_copyfile
    )

    base_dir = tmp_path / "run"
    with pytest.raises(HooksBootstrapError, match="changed while being copied"):
        bootstrap_hooks_tree(base_dir, source, session_id="s-15", log=log)
    assert not (base_dir / "hooks").exists()
    assert list(base_dir.glob(".hooks-staging-*")) == []


def _refuse_chown(path, uid, gid) -> None:
    raise PermissionError(f"operation not permitted: {path} to {uid}:{gid}")


def test_an_ownership_that_cannot_be_set_keeps_the_owner_and_sets_the_mode(
    tmp_path, monkeypatch
) -> None:
    from midrun.hooks.manager.apply_ownership_and_mode import (
        apply_ownership_and_mode,
    )

    target = tmp_path / "file.sh"
    target.write_text("echo x\n")
    target.chmod(0o644)
    before = target.stat()
    monkeypatch.setattr(os, "chown", _refuse_chown)

    preserved = apply_ownership_and_mode(
        target, before.st_uid + 1, before.st_gid, 0o755
    )

    after = target.stat()
    assert preserved is False
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert stat.S_IMODE(after.st_mode) == 0o755


def _refuse_unmapped_chown(path, uid, gid) -> None:
    raise OSError(errno.EINVAL, f"invalid argument: {path} to {uid}:{gid}")


def test_an_ownership_outside_the_user_namespace_keeps_the_owner_and_sets_the_mode(
    tmp_path, monkeypatch
) -> None:
    """In a user namespace, chown to an unmapped uid/gid fails with EINVAL."""
    from midrun.hooks.manager.apply_ownership_and_mode import (
        apply_ownership_and_mode,
    )

    target = tmp_path / "file.sh"
    target.write_text("echo x\n")
    target.chmod(0o644)
    before = target.stat()
    monkeypatch.setattr(os, "chown", _refuse_unmapped_chown)

    preserved = apply_ownership_and_mode(target, 65534, 65534, 0o755)

    after = target.stat()
    assert preserved is False
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert stat.S_IMODE(after.st_mode) == 0o755


def _warnings_of_bootstrap(base_dir, source_dir, session_id) -> list[str]:
    warnings: list[str] = []
    sink_id = logger.add(warnings.append, level="WARNING")
    try:
        bootstrap_hooks_tree(base_dir, source_dir, session_id=session_id, log=logger)
    finally:
        logger.remove(sink_id)
    return warnings


@pytest.mark.parametrize("spelling", ["relative", "symlink"])
def test_a_resume_with_the_same_baseline_spelled_differently_does_not_warn(
    tmp_path, monkeypatch, spelling
) -> None:
    (tmp_path / "baseline" / "light").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "baseline")
    monkeypatch.chdir(tmp_path)
    configured = Path("baseline") if spelling == "relative" else tmp_path / "link"
    base_dir = tmp_path / "run"
    bootstrap_hooks_tree(base_dir, configured, session_id="s-a", log=logger)

    warnings = _warnings_of_bootstrap(base_dir, configured, "s-b")

    assert not [w for w in warnings if "differs" in w]


def test_a_resume_with_another_baseline_warns_that_it_differs(tmp_path) -> None:
    (tmp_path / "baseline" / "light").mkdir(parents=True)
    (tmp_path / "other" / "light").mkdir(parents=True)
    base_dir = tmp_path / "run"
    bootstrap_hooks_tree(base_dir, tmp_path / "baseline", session_id="s-a", log=logger)

    warnings = _warnings_of_bootstrap(base_dir, tmp_path / "other", "s-b")

    assert len([w for w in warnings if "differs" in w]) == 1


def test_a_resume_warns_only_when_the_tree_has_another_catalogue_version(
    tmp_path,
) -> None:
    base_dir = tmp_path / "run"
    bootstrap_hooks_tree(base_dir, None, session_id="s-a", log=logger)
    assert not [
        w for w in _warnings_of_bootstrap(base_dir, None, "s-b") if "catalogue" in w
    ]

    marker_path = base_dir / "midrun-hooks-state.json"
    marker = json.loads(marker_path.read_text())
    marker["catalogue_version"] = CATALOGUE_VERSION - 1
    marker_path.write_text(json.dumps(marker))
    warnings = _warnings_of_bootstrap(base_dir, None, "s-c")

    assert len([w for w in warnings if "catalogue version" in w]) == 1


def test_a_baseline_of_another_group_is_copied_with_one_warning(
    tmp_path, monkeypatch
) -> None:
    """A non-root user cannot give a copy a group they are not a member of."""
    source = tmp_path / "baseline"
    (source / "light").mkdir(parents=True)
    armed = source / "light" / "on_fit_start.sh"
    armed.write_text("#!/usr/bin/env bash\necho armed\n")
    armed.chmod(0o755)
    real_lstat = os.lstat

    def lstat_of_another_group(path, *args, **kwargs):
        st = real_lstat(path, *args, **kwargs)
        if not str(path).startswith(str(source)):
            return st
        extra = {name: getattr(st, name) for name in ("st_mtime_ns", "st_ctime_ns")}
        return os.stat_result((*st[:5], st.st_gid + 1, *st[6:10]), extra)

    monkeypatch.setattr(os, "lstat", lstat_of_another_group)
    monkeypatch.setattr(os, "chown", _refuse_chown)
    warnings: list[str] = []
    sink_id = logger.add(warnings.append, level="WARNING")
    try:
        live_dir = bootstrap_hooks_tree(
            tmp_path / "run", source, session_id="s-16", log=logger
        ).live_dir
    finally:
        logger.remove(sink_id)

    copied = live_dir / "light" / "on_fit_start.sh"
    assert copied.read_text() == armed.read_text()
    assert stat.S_IMODE(copied.stat().st_mode) == 0o755
    ownership_warnings = [w for w in warnings if "ownership" in w]
    assert len(ownership_warnings) == 1
    assert "light/on_fit_start.sh" in ownership_warnings[0]
