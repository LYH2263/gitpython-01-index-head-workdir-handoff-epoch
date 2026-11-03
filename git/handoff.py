# Copyright (C) 2026 Michael Trier (mtrier@gmail.com) and contributors
#
# This module is part of GitPython and is released under the
# 3-Clause BSD License: https://opensource.org/license/bsd-3-clause/

"""Generation-scoped handoff ledgers for the index, the HEAD tip, and the worktree.

This module is opt-in. Unless a :class:`GenerationLedger` scope is active on a
:class:`~git.repo.base.Repo`, every carrier (``IndexFile.add``/``write``/
``write_tree``/``commit``/``checkout``/``reset``, ``HEAD.reset`` and the native
``SymbolicReference.set_reference`` tip write) behaves exactly as it does without
this module.

Within an active generation the three channels keep separate books:

* the *index* channel is redirected to a generation-private index file, so staging
  performed in an old generation can never be materialized by a
  ``write_tree``/``commit`` of a newer generation;
* the *tip* channel records the previous state of ``HEAD``, the active branch ref
  and ``ORIG_HEAD`` before every move, so a generation that fails can roll the tip
  back;
* the *worktree* channel records the prior contents (or absence) of every file a
  generation writes through ``IndexFile.checkout`` or a hard ``HEAD.reset``, so a
  generation that fails can take its files back off disk.

Generations are monotonic per ledger. :meth:`HandoffGeneration.advance` seals the
current generation and opens the next one from the last *committed* index state:
uncommitted staging of the sealed generation is left behind on its own private
ledger. Scopes nest (on the same repository or on a different repository opened by
short-lived code); a nested scope that raises rolls back only its own side effects,
while a nested scope that finishes cleanly hands its result to the enclosing
generation.

Ledgers are shared between :class:`~git.repo.base.Repo` instances that open the same
git directory, so a second short-lived ``Repo`` handle cannot flush the state of an
outer generation through a different in-memory index cache.
"""

__all__ = ["GenerationLedger", "HandoffGeneration"]

import contextlib
import os
import os.path as osp
import weakref
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    TYPE_CHECKING,
    Tuple,
    TypeVar,
    Union,
)

from git.exc import GenerationHandoffError

if TYPE_CHECKING:
    from git.cmd import Git

    from git.repo import Repo

# Refs that live next to HEAD in the per-worktree git directory rather than in the
# common directory; mirrors git.refs.symbolic._git_dir without importing it.
_GIT_DIR_REFS = frozenset({"HEAD", "ORIG_HEAD", "FETCH_HEAD", "index", "logs"})


class _RefState(NamedTuple):
    """Prior on-disk state of a loose ref touched by a generation."""

    rel: str
    abspath: str
    existed: bool
    data: Optional[bytes]


class _WorktreeState(NamedTuple):
    """Prior on-disk state of a worktree path touched by a generation."""

    abspath: str
    existed: bool
    is_symlink: bool
    data: Optional[bytes]


class _ResetEvent(NamedTuple):
    """Which fork of ``HEAD.reset`` a generation actually drove through porcelain."""

    mode: Optional[str]
    paths: Tuple[str, ...]
    moved_tip: bool


def _read_file(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fp:
            return fp.read()
    except OSError:
        return None


def _ref_abspath(repo: "Repo", rel: str) -> str:
    base = repo.git_dir if rel in _GIT_DIR_REFS else repo.common_dir
    return osp.join(os.fspath(base), rel)


_T = TypeVar("_T", _RefState, _WorktreeState)


def _earliest(states: Sequence[_T]) -> Dict[Any, _T]:
    """Keep the first recorded state per key, so scope-entry state wins on restore."""
    earliest: Dict[Any, _T] = {}
    for state in states:
        earliest.setdefault(state[0], state)
    return earliest


class HandoffGeneration:
    """A single generation on a :class:`GenerationLedger`.

    Instances are created by :meth:`GenerationLedger.scope` and
    :meth:`advance`; they are not meant to be constructed directly. Carrier code
    uses the ledger hooks, while hosts use :meth:`advance` to draw a barrier between
    generations.
    """

    __slots__ = (
        "ledger",
        "ordinal",
        "index_path",
        "epoch",
        "ref_undo",
        "worktree_undo",
        "resets",
        "commits",
        "sealed",
        "_worktree_seen",
        "_ref_seen",
    )

    def __init__(self, ledger: "GenerationLedger", ordinal: int, index_path: str) -> None:
        self.ledger = ledger
        self.ordinal = ordinal
        self.index_path = index_path
        #: Bumped when a nested scope republishes our index file, so caches of
        #: long-lived IndexFile instances are forced to reread.
        self.epoch = 0
        self.ref_undo: List[_RefState] = []
        self.worktree_undo: List[_WorktreeState] = []
        self.resets: List[_ResetEvent] = []
        self.commits: List[str] = []
        self.sealed = False
        self._worktree_seen: Set[str] = set()
        self._ref_seen: Set[str] = set()

    def __repr__(self) -> str:
        return "<git.HandoffGeneration g%d %s>" % (self.ordinal, "sealed" if self.sealed else "active")

    @property
    def generation(self) -> int:
        """Monotonic, 1-based ordinal of this generation within its ledger."""
        return self.ordinal

    @property
    def token(self) -> Tuple[int, int]:
        """Opaque cache-invalidation token: ``(ordinal, epoch)``."""
        return (self.ordinal, self.epoch)

    def advance(self) -> "HandoffGeneration":
        """Seal this generation and open the next one on the same scope.

        The next generation is seeded from the last committed index state, so
        staging that was only performed (and not committed) in this generation stays
        on this generation's private ledger and cannot reach a later
        ``write_tree`` or ``commit``.

        :return:
            The new active :class:`HandoffGeneration`.
        """
        return self.ledger.advance(self)

    # { carrier hooks

    @contextlib.contextmanager
    def porcelain_environment(self, git: "Git") -> Iterator[None]:
        """Redirect git porcelain driven by a carrier at this generation's index.

        ``git`` is the :class:`~git.cmd.Git` of the carrier's repository; the
        environment override is restored on exit, so porcelain called by an outer
        generation or by a different repository never inherits it.
        """
        with git.custom_environment(GIT_INDEX_FILE=os.fspath(self.index_path)):
            yield

    def snapshot_ref(self, repo: "Repo", rel: Union[str, "os.PathLike[str]"]) -> None:
        """Record the prior loose-ref state of `rel` before a generation moves it.

        Only the first snapshot per ref within a generation is kept; that is the
        state the generation has to restore if it fails.
        """
        rel_str = os.fspath(rel)
        if rel_str in self._ref_seen:
            return
        self._ref_seen.add(rel_str)
        abspath = _ref_abspath(repo, rel_str)
        data = _read_file(abspath)
        self.ref_undo.append(_RefState(rel_str, abspath, data is not None, data))

    def snapshot_tip_refs(self, repo: "Repo") -> None:
        """Snapshot ``HEAD``, the branch it is attached to, and ``ORIG_HEAD``."""
        self.snapshot_ref(repo, "HEAD")
        self.snapshot_ref(repo, "ORIG_HEAD")
        head_data = _read_file(_ref_abspath(repo, "HEAD"))
        if head_data is not None:
            text = head_data.decode("utf-8", "replace").strip()
            if text.startswith("ref:"):
                self.snapshot_ref(repo, text.split(":", 1)[1].strip())

    def snapshot_worktree(self, repo: "Repo", rel_paths: Sequence[Union[str, "os.PathLike[str]"]]) -> None:
        """Record prior contents (or absence) of worktree paths this gen will write."""
        wtd = repo.working_tree_dir
        if wtd is None:
            return
        wtd_str = os.fspath(wtd)
        for rel in rel_paths:
            rel_str = os.fspath(rel).replace("\\", "/")
            abspath = osp.normpath(osp.join(wtd_str, rel_str))
            if abspath in self._worktree_seen:
                continue
            self._worktree_seen.add(abspath)
            is_symlink = osp.islink(abspath)
            data: Optional[bytes] = None
            existed = False
            if is_symlink:
                existed = True
                data = os.readlink(abspath).encode("utf-8")
            elif osp.isfile(abspath):
                existed = True
                data = _read_file(abspath)
            self.worktree_undo.append(_WorktreeState(abspath, existed, is_symlink, data))

    def record_reset(self, mode: Optional[str], paths: Sequence[str], moved_tip: bool) -> None:
        """Record which fork of ``HEAD.reset`` (soft/mixed/hard/paths) was driven."""
        self.resets.append(_ResetEvent(mode, tuple(paths), moved_tip))

    def note_commit(self, hexsha: str) -> None:
        """Note that ``IndexFile.commit`` published a tree/commit/tip in this gen.

        This rebases the seed of any later generation onto the committed index, so
        a commit (as opposed to a tip-only ``HEAD.reset``) is the cross-generation
        materialization point for the index channel.
        """
        self.commits.append(hexsha)
        self.ledger.rebase_after_commit(self)

    # } END carrier hooks


class _Scope:
    """A stack of generations between ``GenerationLedger.scope`` enter and exit."""

    __slots__ = ("gens", "entry_index_bytes", "baseline_index_bytes", "parent")

    def __init__(
        self,
        entry_index_bytes: Optional[bytes],
        baseline_index_bytes: Optional[bytes],
        parent: Optional[HandoffGeneration],
    ) -> None:
        self.gens: List[HandoffGeneration] = []
        #: Index state to restore if the whole scope fails.
        self.entry_index_bytes = entry_index_bytes
        #: Seed for the next ``advance`` (rebased by commits).
        self.baseline_index_bytes = baseline_index_bytes
        #: The enclosing generation when scopes nest on the same ledger.
        self.parent = parent


class GenerationLedger:
    """Per-repository bookkeeping for handoff generations.

    Obtain the ledger of a repository through :attr:`Repo.generations
    <git.repo.base.Repo.generations>` and open generations with
    :meth:`scope`. Ledgers for the same git directory are shared, so nested
    short-lived :class:`~git.repo.base.Repo` instances participate in the same
    lineage; Repo instances for different repositories never do.
    """

    #: git directory -> ledger, shared while any Repo of that directory is alive.
    _registry: "weakref.WeakValueDictionary[str, GenerationLedger]" = weakref.WeakValueDictionary()

    __slots__ = ("repo", "_key", "_counter", "_scopes", "__weakref__")

    def __init__(self, repo: "Repo", key: str) -> None:
        self.repo = repo
        self._key = key
        self._counter = 0
        self._scopes: List[_Scope] = []

    @classmethod
    def bind(cls, repo: "Repo") -> "GenerationLedger":
        """Return the shared ledger for `repo`'s git directory, creating it once."""
        key = osp.realpath(os.fspath(repo.git_dir))
        ledger = cls._registry.get(key)
        if ledger is None:
            ledger = cls(repo, key)
            cls._registry[key] = ledger
        return ledger

    @property
    def current(self) -> Optional[HandoffGeneration]:
        """:return: The active generation, or ``None`` outside any scope."""
        if not self._scopes:
            return None
        return self._scopes[-1].gens[-1]

    @property
    def generation(self) -> Optional[int]:
        """:return: Ordinal of the active generation, or ``None`` if idle."""
        current = self.current
        return None if current is None else current.ordinal

    @property
    def depth(self) -> int:
        """:return: Number of currently nested scopes on this ledger."""
        return len(self._scopes)

    def _real_index_path(self) -> str:
        return osp.join(os.fspath(self.repo.git_dir), "index")

    def _read_real_index(self) -> Optional[bytes]:
        return _read_file(self._real_index_path())

    def _write_index_file(self, path: str, data: Optional[bytes]) -> None:
        if data is None:
            with contextlib.suppress(OSError):
                os.remove(path)
            return
        directory = osp.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as fp:
            fp.write(data)

    def _seed_generation_file(self, gen: HandoffGeneration, data: Optional[bytes]) -> None:
        # Unique per ordinal, so there is no stale file to remove first.
        if data is not None:
            self._write_index_file(gen.index_path, data)

    def _new_generation(self, scope: _Scope) -> HandoffGeneration:
        self._counter += 1
        ordinal = self._counter
        gen = HandoffGeneration(
            self,
            ordinal,
            osp.join(os.fspath(self.repo.git_dir), "index.handoff-%d" % ordinal),
        )
        self._seed_generation_file(gen, scope.baseline_index_bytes)
        scope.gens.append(gen)
        return gen

    @contextlib.contextmanager
    def scope(self) -> Iterator[HandoffGeneration]:
        """Open a generation scope.

        On clean exit the final generation's index is handed off: to the enclosing
        generation's private index for nested scopes, or to the repository's real
        index for the outermost scope. Tip moves and worktree writes already on disk
        remain in place; only the index channel is published.

        If the body raises, everything the scope changed across all of its
        generations is rolled back (index, tip refs and worktree files), its private
        index files are removed, and the exception propagates.
        """
        parent = self.current
        if parent is None:
            entry_bytes = self._read_real_index()
        else:
            entry_bytes = _read_file(parent.index_path)
        scope_obj = _Scope(
            entry_index_bytes=entry_bytes,
            baseline_index_bytes=entry_bytes,
            parent=parent,
        )
        gen = self._new_generation(scope_obj)
        self._scopes.append(scope_obj)
        try:
            yield gen
        except BaseException:
            self._rollback_scope(scope_obj)
            raise
        else:
            self._handoff_scope(scope_obj)
        finally:
            self._discard_scope_files(scope_obj)
            self._scopes.pop()

    def advance(self, gen: HandoffGeneration) -> HandoffGeneration:
        if not self._scopes or self._scopes[-1].gens[-1] is not gen:
            raise GenerationHandoffError("only the active generation can be advanced")
        if gen.sealed:
            raise GenerationHandoffError("generation %d was already advanced" % gen.ordinal)
        scope_obj = self._scopes[-1]
        gen.sealed = True
        return self._new_generation(scope_obj)

    def rebase_after_commit(self, gen: HandoffGeneration) -> None:
        for scope_obj in self._scopes:
            if gen in scope_obj.gens:
                scope_obj.baseline_index_bytes = _read_file(gen.index_path)
                return

    def _scope_undo(self, scope_obj: _Scope) -> Tuple[List[_RefState], List[_WorktreeState]]:
        refs: List[_RefState] = []
        worktree: List[_WorktreeState] = []
        for g in scope_obj.gens:
            refs.extend(g.ref_undo)
            worktree.extend(g.worktree_undo)
        return refs, worktree

    def _restore_refs(self, refs: Sequence[_RefState]) -> None:
        for state in _earliest(refs).values():
            with contextlib.suppress(OSError):
                if state.existed and state.data is not None:
                    directory = osp.dirname(state.abspath)
                    if directory:
                        os.makedirs(directory, exist_ok=True)
                    with open(state.abspath, "wb") as fp:
                        fp.write(state.data)
                else:
                    os.remove(state.abspath)

    def _restore_worktree(self, states: Sequence[_WorktreeState]) -> None:
        for state in _earliest(states).values():
            with contextlib.suppress(OSError):
                if not state.existed:
                    os.remove(state.abspath)
                elif state.is_symlink:
                    with contextlib.suppress(OSError):
                        os.remove(state.abspath)
                    target = os.fsdecode(state.data) if state.data is not None else ""
                    os.symlink(target, state.abspath)
                elif state.data is not None:
                    directory = osp.dirname(state.abspath)
                    if directory:
                        os.makedirs(directory, exist_ok=True)
                    with open(state.abspath, "wb") as fp:
                        fp.write(state.data)

    def _rollback_scope(self, scope_obj: _Scope) -> None:
        refs, worktree = self._scope_undo(scope_obj)
        # Restore the tip first, then the index and the worktree, mirroring the order
        # in which the channels diverged. Best-effort: never mask the original error
        # with a restore failure on a locked file.
        self._restore_refs(refs)
        if scope_obj.parent is None:
            with contextlib.suppress(OSError):
                self._write_index_file(self._real_index_path(), scope_obj.entry_index_bytes)
        # A nested scope never wrote the parent generation's index file; leave it.
        self._restore_worktree(worktree)

    def _handoff_scope(self, scope_obj: _Scope) -> None:
        final_gen = scope_obj.gens[-1]
        final_bytes = _read_file(final_gen.index_path)
        if scope_obj.parent is None:
            with contextlib.suppress(OSError):
                self._write_index_file(self._real_index_path(), final_bytes)
        else:
            parent = scope_obj.parent
            with contextlib.suppress(OSError):
                self._write_index_file(parent.index_path, final_bytes)
            # The enclosing generation now owns the nested scope's effects: merge its
            # undo logs (earliest snapshots already present stay first) and bump its
            # epoch so stale index caches reread the republished index file.
            refs, worktree = self._scope_undo(scope_obj)
            parent.ref_undo.extend(refs)
            parent._ref_seen.update(state.rel for state in refs)
            parent.worktree_undo.extend(worktree)
            parent._worktree_seen.update(state.abspath for state in worktree)
            for inner_gen in scope_obj.gens:
                parent.commits.extend(inner_gen.commits)
                parent.resets.extend(inner_gen.resets)
            parent.epoch += 1
            # If the nested scope committed, its last-committed index is the new
            # barrier seed for the enclosing scope, so a later advance() there does
            # not resurrect the pre-inner-commit index state. Staging-only nested
            # scopes leave the enclosing barrier untouched.
            if any(inner_gen.commits for inner_gen in scope_obj.gens):
                for enclosing in reversed(self._scopes[:-1]):
                    if parent in enclosing.gens:
                        enclosing.baseline_index_bytes = scope_obj.baseline_index_bytes
                        break

    def _discard_scope_files(self, scope_obj: _Scope) -> None:
        for gen in scope_obj.gens:
            for suffix in ("", ".lock"):
                with contextlib.suppress(OSError):
                    os.remove(gen.index_path + suffix)
