# Copyright (C) 2026 Michael Trier (mtrier@gmail.com) and contributors
#
# This module is part of GitPython and is released under the
# 3-Clause BSD License: https://opensource.org/license/bsd-3-clause/

"""Acceptance tests for index/HEAD/worktree handoff generations.

These tests exercise *causal chains across the different command exits*:
index staging versus ``write_tree``/``commit`` materialization, tip-only
``HEAD.reset`` (soft) versus index/worktree forks (hard/paths), and
``IndexFile.checkout`` worktree writes versus ``IndexFile.commit`` tip moves.
They deliberately do not enumerate a soft/mixed/hard reset matrix.

Every test builds a throwaway repository in a temporary directory; no fixture
repository or remote service is used.
"""

import glob
import os
import os.path as osp

import pytest

from git import GenerationHandoffError, GenerationLedger, HandoffGeneration, Repo

# -----------------------------------------------------------------------------
# Temporary-repository helpers
# -----------------------------------------------------------------------------


def _init_repo(path) -> Repo:
    repo = Repo.init(path)
    with repo.config_writer() as writer:
        writer.set_value("user", "name", "Handoff Tester")
        writer.set_value("user", "email", "handoff@example.com")
    return repo


def _write(repo: Repo, name: str, text: str) -> str:
    path = osp.join(str(repo.working_tree_dir), *name.split("/"))
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(text)
    return path


def _read(repo: Repo, name: str) -> str:
    with open(osp.join(str(repo.working_tree_dir), *name.split("/")), encoding="utf-8") as stream:
        return stream.read()


def _exists(repo: Repo, name: str) -> bool:
    return osp.exists(osp.join(str(repo.working_tree_dir), *name.split("/")))


def _commit_files(repo: Repo, files, message: str):
    """Create (name, text) files, stage them and create a commit."""
    for name, text in files:
        _write(repo, name, text)
    repo.index.add([name for name, _ in files])
    return repo.index.commit(message)


def _entry_names(repo: Repo):
    return sorted(entry.path for entry in repo.index.entries.values())


def _tree_names(tree) -> list:
    return sorted(blob.path for blob in tree.blobs)


def _handoff_files(repo: Repo) -> list:
    return glob.glob(osp.join(str(repo.git_dir), "index.handoff-*"))


def _real_index_bytes(repo: Repo) -> bytes:
    with open(osp.join(str(repo.git_dir), "index"), "rb") as stream:
        return stream.read()


@pytest.fixture
def repo(tmp_path) -> Repo:
    repository = _init_repo(tmp_path)
    yield repository
    repository.close()


class _Boom(RuntimeError):
    """Sentinel failure raised from inside generation bodies."""


# -----------------------------------------------------------------------------
# Pin parity
# -----------------------------------------------------------------------------


def test_idle_repo_has_no_generation_and_no_index_redirect(repo: Repo) -> None:
    assert repo.generations.current is None
    assert repo.generations.generation is None
    assert "GIT_INDEX_FILE" not in repo.git.environment()
    assert isinstance(repo.generations, GenerationLedger)


def test_operations_without_generation_never_touch_private_ledger(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a")], "c0")
    _write(repo, "b.txt", "b")
    repo.index.add(["b.txt"])
    repo.index.commit("c1")
    repo.head.reset(c0, working_tree=True)
    repo.index.checkout(["a.txt"])
    assert repo.generations.current is None
    assert _handoff_files(repo) == []


def test_porcelain_inside_scope_but_outside_carrier_sees_real_index(repo: Repo) -> None:
    _commit_files(repo, [("a.txt", "a")], "c0")
    with repo.generation_scope():
        _write(repo, "b.txt", "b")
        repo.index.add(["b.txt"])
        # A raw porcelain call is not a carrier and must not inherit the
        # generation's GIT_INDEX_FILE redirect.
        staged = repo.git.diff("--cached", "--name-only").splitlines()
        assert staged == []
        assert "GIT_INDEX_FILE" not in repo.git.environment()


# -----------------------------------------------------------------------------
# Chain 1: the staging barrier at an advance()
# -----------------------------------------------------------------------------


def test_staging_in_old_generation_cannot_reach_new_write_tree_or_commit(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a")], "c0")
    with repo.generation_scope() as first:
        _write(repo, "b.txt", "b")
        repo.index.add(["b.txt"])
        assert _entry_names(repo) == ["a.txt", "b.txt"]
        # Within the generation the staging is materializable.
        assert _tree_names(repo.index.write_tree()) == ["a.txt", "b.txt"]

        second = first.advance()
        assert isinstance(second, HandoffGeneration)
        assert second.generation == first.generation + 1

        # Sealed staging must not leak into the new generation's index, tree
        # or commit, even though b.txt still exists on disk.
        assert _entry_names(repo) == ["a.txt"]
        assert _tree_names(repo.index.write_tree()) == ["a.txt"]
        new_commit = repo.index.commit("gen two")
        assert _tree_names(new_commit.tree) == ["a.txt"]
        assert new_commit.parents == [c0]
        assert repo.head.commit == new_commit
    # Clean handoff publishes the last generation's index (without b.txt).
    assert _entry_names(repo) == ["a.txt"]


def test_committed_index_is_the_seed_of_the_next_generation(repo: Repo) -> None:
    _commit_files(repo, [("a.txt", "a")], "c0")
    with repo.generation_scope() as first:
        _write(repo, "c.txt", "c")
        repo.index.add(["c.txt"])
        committed = repo.index.commit("with c")
        second = first.advance()
        assert second.generation == first.generation + 1
        assert _entry_names(repo) == ["a.txt", "c.txt"]
        assert repo.index.entries[("c.txt", 0)].hexsha == committed.tree["c.txt"].hexsha


def test_real_index_and_tip_are_untouched_while_scope_is_open(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a")], "c0")
    real_index = _real_index_bytes(repo)
    with repo.generation_scope():
        _write(repo, "b.txt", "b")
        repo.index.add(["b.txt"])
        repo.index.write()
        assert _real_index_bytes(repo) == real_index
        assert repo.head.commit == c0


# -----------------------------------------------------------------------------
# Chain 2: HEAD.reset forks (soft / hard / paths) at the generation boundary
# -----------------------------------------------------------------------------


def test_soft_reset_failure_restores_tip_but_keeps_index_and_worktree(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a")], "c0")
    c1 = _commit_files(repo, [("b.txt", "b")], "c1")
    with pytest.raises(_Boom):
        with repo.generation_scope() as generation:
            repo.head.reset(c0, index=False)
            assert repo.head.commit == c0
            # Soft moves the tip only: index still has both files.
            assert _entry_names(repo) == ["a.txt", "b.txt"]
            assert _exists(repo, "b.txt")
            assert generation.resets[-1].mode == "--soft"
            assert generation.resets[-1].moved_tip is True
            raise _Boom()
    assert repo.head.commit == c1
    assert _entry_names(repo) == ["a.txt", "b.txt"]
    assert _read(repo, "b.txt") == "b"


def test_hard_reset_failure_restores_tip_index_and_worktree(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a0")], "c0")
    c1 = _commit_files(repo, [("a.txt", "a1"), ("b.txt", "b")], "c1")
    # An unstaged worktree modification that exists before the generation.
    _write(repo, "a.txt", "dirty")
    with pytest.raises(_Boom):
        with repo.generation_scope() as generation:
            repo.head.reset(c0, working_tree=True)
            assert repo.head.commit == c0
            assert _entry_names(repo) == ["a.txt"]
            assert _read(repo, "a.txt") == "a0"
            assert not _exists(repo, "b.txt")
            assert generation.resets[-1].mode == "--hard"
            raise _Boom()
    assert repo.head.commit == c1
    assert _entry_names(repo) == ["a.txt", "b.txt"]
    # The dirty worktree contents and the removed file both come back.
    assert _read(repo, "a.txt") == "dirty"
    assert _read(repo, "b.txt") == "b"


def test_paths_reset_does_not_move_tip_and_is_sealed_behind_advance(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a"), ("p.txt", "p0")], "c0")
    c1 = _commit_files(repo, [("p.txt", "p1")], "c1")
    with repo.generation_scope() as first:
        repo.head.reset(c0, paths=["p.txt"])
        # Paths-limited reset never moves the tip.
        assert repo.head.commit == c1
        assert first.resets[-1].moved_tip is False
        assert first.resets[-1].paths == ("p.txt",)
        # ... but the private generation index carries the partial reset.
        assert repo.index.entries[("p.txt", 0)].hexsha == c0.tree["p.txt"].hexsha

        first.advance()
        # The partial reset was uncommitted: the new generation is seeded from
        # the committed index, so p.txt is back at its c1 state.
        assert repo.index.entries[("p.txt", 0)].hexsha == c1.tree["p.txt"].hexsha
        sealed_commit = repo.index.commit("after barrier")
        assert sealed_commit.tree.hexsha == c1.tree.hexsha
        assert repo.head.commit == sealed_commit
    assert repo.head.commit == sealed_commit


def test_paths_reset_failure_restores_index_without_touching_tip_or_worktree(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a"), ("p.txt", "p0")], "c0")
    c1 = _commit_files(repo, [("p.txt", "p1")], "c1")
    worktree_before = _read(repo, "p.txt")
    with pytest.raises(_Boom):
        with repo.generation_scope():
            repo.head.reset(c0, paths=["p.txt"])
            raise _Boom()
    assert repo.head.commit == c1
    assert repo.index.entries[("p.txt", 0)].hexsha == c1.tree["p.txt"].hexsha
    assert _read(repo, "p.txt") == worktree_before


# -----------------------------------------------------------------------------
# Chain 3: IndexFile.checkout (worktree) forks from IndexFile.commit (tip)
# -----------------------------------------------------------------------------


def test_checkout_writes_worktree_without_moving_tip_and_failure_takes_file_back(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a")], "c0")
    with pytest.raises(_Boom):
        with repo.generation_scope():
            os.remove(osp.join(str(repo.working_tree_dir), "a.txt"))
            returned = repo.index.checkout(["a.txt"])
            assert list(returned) == ["a.txt"]
            assert _read(repo, "a.txt") == "a"
            # checkout is not a tip move.
            assert repo.head.commit == c0
            raise _Boom()
    assert repo.head.commit == c0
    assert not _exists(repo, "a.txt")


def test_host_authored_file_survives_failure_but_its_staging_does_not(repo: Repo) -> None:
    _commit_files(repo, [("a.txt", "a")], "c0")
    with pytest.raises(_Boom):
        with repo.generation_scope():
            _write(repo, "host.txt", "host")
            repo.index.add(["host.txt"])
            assert "host.txt" in _entry_names(repo)
            raise _Boom()
    # The worktree channel only undoes carrier writes: the host file remains
    # on disk, but the staging is rolled back off the real index.
    assert _read(repo, "host.txt") == "host"
    assert _entry_names(repo) == ["a.txt"]


def test_commit_moves_tip_and_failure_moves_it_back(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a")], "c0")
    with pytest.raises(_Boom):
        with repo.generation_scope():
            _write(repo, "b.txt", "b")
            repo.index.add(["b.txt"])
            made = repo.index.commit("gen commit")
            assert repo.head.commit == made
            assert _tree_names(made.tree) == ["a.txt", "b.txt"]
            raise _Boom()
    assert repo.head.commit == c0
    assert _entry_names(repo) == ["a.txt"]


def test_clean_checkout_scope_hands_file_off(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a")], "c0")
    with repo.generation_scope():
        os.remove(osp.join(str(repo.working_tree_dir), "a.txt"))
        repo.index.checkout(["a.txt"])
    assert _read(repo, "a.txt") == "a"
    assert repo.head.commit == c0


def test_native_index_reset_with_head_and_worktree_rolls_back(repo: Repo) -> None:
    c0 = _commit_files(repo, [("a.txt", "a0")], "c0")
    c1 = _commit_files(repo, [("a.txt", "a1")], "c1")
    with pytest.raises(_Boom):
        with repo.generation_scope():
            repo.index.reset(c0, head=True, working_tree=True)
            assert repo.head.commit == c0
            assert _read(repo, "a.txt") == "a0"
            raise _Boom()
    assert repo.head.commit == c1
    assert _read(repo, "a.txt") == "a1"


def test_git_rm_and_mv_worktree_writes_roll_back(repo: Repo) -> None:
    _commit_files(repo, [("a.txt", "a"), ("p.txt", "p")], "c0")
    with pytest.raises(_Boom):
        with repo.generation_scope():
            repo.index.remove(["p.txt"], working_tree=True)
            assert not _exists(repo, "p.txt")
            repo.index.move(["a.txt", "renamed.txt"])
            assert _read(repo, "renamed.txt") == "a"
            assert not _exists(repo, "a.txt")
            raise _Boom()
    assert _read(repo, "p.txt") == "p"
    assert _read(repo, "a.txt") == "a"
    assert not _exists(repo, "renamed.txt")
    assert _entry_names(repo) == ["a.txt", "p.txt"]


# -----------------------------------------------------------------------------
# Chain 4: nesting never crosses repository or generation boundaries
# -----------------------------------------------------------------------------


def test_inner_repo_failure_cannot_pollute_outer_generation(tmp_path) -> None:
    outer = _init_repo(osp.join(str(tmp_path), "outer"))
    try:
        c0 = _commit_files(outer, [("a.txt", "a")], "c0")
        with outer.generation_scope():
            _write(outer, "outer.txt", "outer")
            outer.index.add(["outer.txt"])  # Uncommitted outer staging.

            inner_dir = osp.join(str(tmp_path), "inner")
            inner = _init_repo(inner_dir)
            with pytest.raises(_Boom):
                with inner.generation_scope():
                    _write(inner, "i.txt", "i")
                    inner.index.add(["i.txt"])
                    inner.index.commit("inner")
                    assert inner.head.commit.parents == ()
                    raise _Boom()
            # The inner repository is back to its unborn entry state.
            assert list(inner.heads) == []
            assert not osp.exists(osp.join(inner.git_dir, "index"))
            assert inner.generations.depth == 0
            inner.close()

            # The outer lineage is untouched: tip at c0, private staging intact.
            assert outer.head.commit == c0
            assert _entry_names(outer) == ["a.txt", "outer.txt"]
        # Clean outer handoff publishes only the outer staging.
        assert _entry_names(outer) == ["a.txt", "outer.txt"]
        assert outer.head.commit == c0
        assert _handoff_files(outer) == []
    finally:
        outer.close()


def test_second_repo_handle_shares_the_ledger_and_inner_failure_pops_to_outer(repo: Repo) -> None:
    _commit_files(repo, [("a.txt", "a")], "c0")
    handle = Repo(repo.working_tree_dir)
    try:
        with repo.generation_scope():
            _write(repo, "outer.txt", "outer")
            repo.index.add(["outer.txt"])
            with pytest.raises(_Boom):
                with handle.generation_scope():
                    _write(repo, "inner.txt", "inner")
                    handle.index.add(["inner.txt"])
                    assert "inner.txt" in _entry_names(repo)
                    raise _Boom()
            # The inner staging is gone; the outer generation continues.
            assert repo.generations.depth == 1
            assert _entry_names(repo) == ["a.txt", "outer.txt"]
        assert _entry_names(repo) == ["a.txt", "outer.txt"]
    finally:
        handle.close()


def test_inner_commit_is_rolled_back_when_outer_generation_fails(repo: Repo) -> None:
    base = _commit_files(repo, [("a.txt", "a")], "base")
    with pytest.raises(_Boom):
        with repo.generation_scope():
            with repo.generation_scope():
                _write(repo, "i.txt", "i")
                repo.index.add(["i.txt"])
                inner_commit = repo.index.commit("inner")
                assert repo.head.commit == inner_commit
            # A clean inner scope hands the commit to the outer generation.
            assert repo.head.commit == inner_commit
            raise _Boom()
    assert repo.head.commit == base
    assert _entry_names(repo) == ["a.txt"]


def test_clean_committed_inner_scope_rebases_outer_barrier(repo: Repo) -> None:
    _commit_files(repo, [("a.txt", "a")], "c0")
    with repo.generation_scope() as outer:
        with repo.generation_scope():
            _write(repo, "i.txt", "i")
            repo.index.add(["i.txt"])
            repo.index.commit("inner")
        # The outer generation advances after the inner committed handoff: the
        # committed inner index must seed the new outer generation.
        outer.advance()
        assert _entry_names(repo) == ["a.txt", "i.txt"]
    assert _entry_names(repo) == ["a.txt", "i.txt"]
    assert _read(repo, "i.txt") == "i"


def test_private_generation_files_are_removed_in_all_exit_shapes(repo: Repo) -> None:
    _commit_files(repo, [("a.txt", "a")], "c0")
    with repo.generation_scope() as first:
        first.advance()
        with repo.generation_scope():
            pass
        with pytest.raises(_Boom):
            with repo.generation_scope():
                raise _Boom()
    assert _handoff_files(repo) == []


# -----------------------------------------------------------------------------
# Advance guards
# -----------------------------------------------------------------------------


def test_advance_rejects_sealed_and_inactive_generations(repo: Repo) -> None:
    _commit_files(repo, [("a.txt", "a")], "c0")
    with repo.generation_scope() as first:
        second = first.advance()
        with pytest.raises(GenerationHandoffError):
            first.advance()
        third = second.advance()
        with pytest.raises(GenerationHandoffError):
            second.advance()
        assert third.generation == 3
