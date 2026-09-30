"""CodeRabbit's recorded comments, located on the PR's merge commit."""
from __future__ import annotations

from dataclasses import dataclass

from arena.corpus import BotComment, Pr
from arena.diffmap import map_line, touches
from arena.gitrepo import Git
from arena.score import Finding


@dataclass(frozen=True)
class Located:
    finding: Finding
    #: The flagged lines were edited later in the PR (the in-PR "acted on" signal,
    #: biased toward the bot: developers only saw its comments).
    acted_on: bool


def locate(git: Git, pr: Pr) -> tuple[list[Located], int]:
    """(comments located on the merge commit, comments dropped because their file is gone)."""
    located: list[Located] = []
    dropped = 0
    for comment in pr.comments:
        result = _locate(git, pr.merge_sha, comment)
        if result is None:
            dropped += 1
        else:
            located.append(result)
    return located, dropped


def _locate(git: Git, merge_sha: str, comment: BotComment) -> Located | None:
    """None when the comment cannot be placed: its commit is gone or its file no longer exists."""
    if not git.has_commit(comment.commit) or not git.exists(merge_sha, comment.path):
        return None
    hunks = git.hunks(comment.commit, merge_sha, comment.path)
    line, _ = map_line(hunks, comment.end)
    return Located(Finding(comment.path, line), touches(hunks, comment.start, comment.end))
