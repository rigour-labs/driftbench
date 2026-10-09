import json

from bench.release_check import check_paths, main


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_ids_hashes_and_anchors_pass(tmp_path):
    corpus = {"prs": [{"number": 1, "reviews": [{"id": 2, "body_sha256": "a" * 64, "body_chars": 10}],
                       "comments": [{"path": "src/app.py", "line": 3}]}],
              "summary": {"kept_by_kind": {"body": 10, "inline": 35}}}
    assert check_paths([write(tmp_path / "c.json", corpus)]) == []


def test_text_keys_and_long_strings_fail(tmp_path):
    leaked = {"prs": [{"reviews": [{"body": "please fix"}], "note": "x" * 401}]}
    found = check_paths([write(tmp_path / "c.json", leaked)])
    assert any("reviews[0].body" in f for f in found) and any("401-character" in f for f in found)


def test_run_records_allow_only_short_messages_and_no_raw_output(tmp_path):
    assert check_paths([write(tmp_path / "ok.json", {"findings": [{"message": "m" * 80, "path": "a.py"}]})]) == []
    long = check_paths([write(tmp_path / "long.json", {"findings": [{"message": "m" * 81}]})])
    assert long and "81 characters (limit 80)" in long[0]
    assert check_paths([write(tmp_path / "raw.json", {"raw": "tool output"})])


def test_cli_scans_directories(tmp_path):
    (tmp_path / "d").mkdir()
    write(tmp_path / "d" / "x.json", {"diff": "--- a\n+++ b"})
    assert main([str(tmp_path / "d")]) == 1
    assert main([str(write(tmp_path / "ok.json", {"id": 1}))]) == 0


def test_scratch_is_skipped_and_an_undecodable_file_is_a_problem_not_a_crash(tmp_path):
    from bench.release_check import check_paths
    run = tmp_path / "run"
    (run / "_scratch" / "home").mkdir(parents=True)
    (run / "_scratch" / "home" / "cache.json").write_bytes(b"\xff\xfe binary")
    (run / "tool").mkdir()
    (run / "tool" / "ok.json").write_text('{"verdict": "pass"}')
    assert check_paths([run]) == []
    (run / "tool" / "bad.json").write_bytes(b"\xff\xfe binary")
    found = check_paths([run])
    assert len(found) == 1 and "bad.json: unreadable (UnicodeDecodeError)" in found[0]
