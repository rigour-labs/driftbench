import subprocess

from bench.report.ordering import labels_precede_run

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@x.test"]


def commit_at(repo, name, text, when):
    (repo / name).write_text(text, encoding="utf-8")
    subprocess.run([*GIT, "-C", str(repo), "add", name], check=True)
    env = {"GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", name], check=True, env=env)


def test_labels_must_be_committed_before_the_run(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    labels = tmp_path / "o__r.yaml"
    commit_at(tmp_path, "o__r.yaml", "a", "2026-10-01T00:00:00+00:00")
    assert labels_precede_run([labels, tmp_path / "missing.yaml"], "2026-10-08T00:00:00+00:00") == (True, "")
    commit_at(tmp_path, "o__r.yaml", "b", "2026-10-09T00:00:00+00:00")
    ok, reason = labels_precede_run([labels], "2026-10-08T00:00:00+00:00")
    assert not ok and "after the run started" in reason
    labels.write_text("edited", encoding="utf-8")
    assert labels_precede_run([labels], "2026-10-10T00:00:00+00:00") == (False, "o__r.yaml has uncommitted changes")
    assert labels_precede_run([labels], "")[0] is False
