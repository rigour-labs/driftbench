"""The heads of the sample, each with its pull request's cutoff and the diff the reviewer would read.

The cutoff is the pull request's `created_at`: every head of one pull request
shares one store, and nothing from the pull request itself (any round) can be
in it. Main as the learner sees it is the last first-parent commit on the
default branch before the cutoff, so no later fix or revert is ever read. The diff is `git diff <merge base>...<head>`, the one `rigour review
--base <merge base>` reads.
"""
from __future__ import annotations

from pathlib import Path

from bench.harness.gitrepo import GitError, RepoCheckout
from bench.subsample import pr_heads


def cutoffs(corpus: dict) -> list[dict]:
    """[{pr, cutoff, base_sha, heads}] for every pull request in a (restricted) corpus."""
    return [{"pr": pr["number"], "cutoff": pr["created_at"], "base_sha": pr["base_sha"], "heads": pr_heads(pr)}
            for pr in corpus["prs"]]


def write_diff(checkout: RepoCheckout, pr: dict, head: str, out: Path) -> str | None:
    """The head's diff written to `out`, or why it could not be made."""
    for sha in (pr["base_sha"], head):
        if not checkout.ensure_commit(sha, pr["pr"]):
            return f"commit {sha[:12]} unavailable"
    try:
        base = checkout.merge_base(pr["base_sha"], head)
        diff = checkout.git("diff", "--no-color", f"{base}...{head}").stdout
    except GitError as exc:
        return str(exc)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(diff, encoding="utf-8")
    return None


def main_before(checkout: RepoCheckout, cutoff: str) -> str:
    """The last first-parent commit on the default branch committed before the cutoff."""
    sha = checkout.git("rev-list", "-1", "--first-parent", f"--before={cutoff}", "origin/HEAD").stdout.strip()
    if not sha:
        raise GitError(f"no commit on the default branch before {cutoff}")
    return sha


def prepare(checkout: RepoCheckout, corpus: dict, diffs: Path) -> list[dict]:
    """Every pull request's cutoff, main as of the cutoff, and heads, each head with its diff file or its error."""
    checkout.ensure_clone()
    prepared = []
    for pr in cutoffs(corpus):
        pr["main_ref"] = main_before(checkout, pr["cutoff"])
        heads = []
        for head in pr["heads"]:
            path = diffs / f"{head}.diff"
            error = write_diff(checkout, pr, head, path)
            heads.append({"sha": head, **({"error": error} if error else {"diff": str(path)})})
        prepared.append({**pr, "heads": heads})
    return prepared
