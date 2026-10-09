import argparse
import subprocess

from bench.labels.fingerprint import label_fingerprint
from bench.labels.sample import draw_sample, sample_path, write_sample
from bench.report.cli import withheld_reason
from tests.git_fixture import commit_file

POINTS = {"repo": "o/r", "points": [{"id": "a", "kind": "inline", "scorable": True,
                                     "anchor": {"path": "x.py", "line": 3, "side": "RIGHT", "commit_sha": "c"}}]}


def workspace(tmp_path):
    repo = tmp_path / "repo"
    (repo / "labels").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    commit_file(repo, "labels/o__r.yaml", "repo: o/r\npoints: {}\n", "labels")
    write_sample(draw_sample(POINTS, 5, seed=1), sample_path(repo / "labels", "o/r"), replace=False)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@x.test", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@x.test", "-C", str(repo), "commit", "-qm", "s"],
                   check=True)
    return argparse.Namespace(labels=repo / "labels"), repo


def test_reasons_in_order(tmp_path):
    args, repo = workspace(tmp_path)
    manifest = {"labels": label_fingerprint(args.labels)}
    assert withheld_reason(args, "o/r", manifest, POINTS) == ""
    assert "no run.json" in withheld_reason(args, "o/r", None, POINTS)
    assert withheld_reason(args, "x/y", manifest, POINTS) == "no labelled sample for this repository"
    other = {"repo": "o/r", "points": [{**POINTS["points"][0], "id": "b"}]}
    assert "different points file" in withheld_reason(args, "o/r", manifest, other)
    commit_file(repo, "labels/o__r.yaml", "repo: o/r\npoints: {a: {label: judgment}}\n", "late label")
    assert "fixed when the run started" in withheld_reason(args, "o/r", manifest, POINTS)
