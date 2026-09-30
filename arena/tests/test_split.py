import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_every_arena_corpus_is_an_eval_repo_and_never_training_data():
    training = json.loads((ROOT / "rlaif" / "repos_training.json").read_text())
    eval_repos = {r.lower() for r in training["_eval_repos_DO_NOT_ADD"]}
    train_repos = {r.lower() for r in training["repos"]}
    arena = {json.loads(p.read_text())["repo"].lower() for p in (ROOT / "arena" / "corpora").glob("*.json")}
    assert arena <= eval_repos, f"arena repos missing from the eval list: {sorted(arena - eval_repos)}"
    assert not arena & train_repos, f"arena repos used for training: {sorted(arena & train_repos)}"
