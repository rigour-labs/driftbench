import subprocess

from bench.labels.fingerprint import label_fingerprint, labels_unchanged
from tests.git_fixture import commit_file


def setup_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "labels").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    commit_file(repo, "labels/o__r.yaml", "points: {a: {label: judgment}}\n", "labels")
    commit_file(repo, "labels/o__r.sample.yaml", "point_ids: [a]\n", "sample")
    return repo, repo / "labels"


def test_unchanged_labels_pass(tmp_path):
    _, labels = setup_repo(tmp_path)
    assert labels_unchanged(labels, label_fingerprint(labels), "o/r") == (True, "")


def test_a_label_changed_after_the_run_started_is_refused_even_if_committed(tmp_path):
    repo, labels = setup_repo(tmp_path)
    recorded = label_fingerprint(labels)
    commit_file(repo, "labels/o__r.yaml", "points: {a: {label: mechanical}}\n", "relabel after the run")
    ok, reason = labels_unchanged(labels, recorded, "o/r")
    assert not ok and "differs from the version fixed when the run started" in reason


def test_dirty_missing_or_unrecorded_labels_are_refused(tmp_path):
    _, labels = setup_repo(tmp_path)
    recorded = label_fingerprint(labels)
    (labels / "o__r.yaml").write_text("edited", encoding="utf-8")
    assert labels_unchanged(labels, recorded, "o/r") == (False, "o__r.yaml has uncommitted changes")
    assert labels_unchanged(labels, None, "o/r")[0] is False
    assert labels_unchanged(labels, recorded, "x/y") == (False, "x__y.yaml is missing")
