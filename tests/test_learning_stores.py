"""bench/learning/stores.mjs: the fetch the learner reads through, and the judge limits it serves with."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STORES = Path(__file__).resolve().parent.parent / "bench" / "learning" / "stores.mjs"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def node(script: str) -> object:
    code = f"const m = await import({json.dumps(STORES.as_uri())});\n{script}"
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    try:
        return json.loads(out.stdout)
    except ValueError as exc:
        pytest.fail(f"node printed no JSON: {exc}: {out.stdout[:200]}")


CRAWL = {"repo": "o/r", "prs": [
    {"number": 1, "merged_at": "2026-01-05T00:00:00Z"}, {"number": 2, "merged_at": "2026-01-10T00:00:00Z"},
    {"number": 4, "merged_at": "2026-01-22T00:00:00Z"}],
    "reviews": {"1": {"comments": [{"id": 12, "created_at": "2026-01-25T00:00:00Z"}, {"id": 11, "created_at": "2026-01-03T00:00:00Z"}],
                      "reviews": [{"id": 7, "submitted_at": "2026-01-30T00:00:00Z"}, {"id": 8, "submitted_at": None}]}}}


def test_the_learner_sees_only_what_existed_before_the_cutoff():
    got = node(f"""
const f = m.cachedFetch({json.dumps(CRAWL)}, '2026-01-20T00:00:00Z');
const read = async (u) => (await f(u)).json();
const base = 'https://cached.invalid/repos/o/r';
let refused = '';
try {{ await f('https://api.github.com/repos/o/r/pulls'); }} catch (e) {{ refused = e.message; }}
console.log(JSON.stringify({{
  list: (await read(base + '/pulls?state=closed&page=1')).map(p => p.number),
  page2: (await read(base + '/pulls?state=closed&page=2')).length,
  comments: (await read(base + '/pulls/1/comments?per_page=100')).map(c => c.id),
  reviews: (await read(base + '/pulls/1/reviews?per_page=100')).length,
  refused }}));""")
    assert got["list"] == [2, 1] and got["page2"] == 0          # #4 merged after the cutoff; newest merged first
    assert got["comments"] == [11] and got["reviews"] == 0       # written after the cutoff (or never submitted): unseen
    assert "unexpected request" in got["refused"]                # nothing reaches the real API


def test_the_judge_limits_come_from_the_installed_reviewer_or_the_precheck_refuses():
    source = ("const JUDGE_STANDARDS = 15;\nconst JUDGE_FILE_LESSONS = 30;\nconst JUDGE_LESSONS_PER_FILE = 3;\n"
              "lessonsForDiff(input.cwd, input.diff, input.lessons, JUDGE_STANDARDS, JUDGE_FILE_LESSONS, JUDGE_LESSONS_PER_FILE, input.pr)")
    got = node(f"""
let changed = '';
try {{ m.judgeLimits({json.dumps(source.replace("JUDGE_FILE_LESSONS, JUDGE", "LIMIT, JUDGE"))}); }} catch (e) {{ changed = e.message; }}
console.log(JSON.stringify({{ limits: m.judgeLimits({json.dumps(source)}), changed }}));""")
    assert got["limits"] == {"standards": 15, "limit": 30, "perFile": 3}
    assert "no longer calls lessonsForDiff" in got["changed"]


def test_comments_and_reviews_are_paged_exactly_with_none_repeated_or_missing():
    comments = [{"id": i, "created_at": f"2026-01-0{1 + i % 9}T{i % 24:02d}:{i % 60:02d}:00Z"} for i in range(150)]
    crawl = {"repo": "o/r", "prs": [], "reviews": {"1": {"comments": comments, "reviews": []}}}
    got = node(f"""
const f = m.cachedFetch({json.dumps(crawl)}, '2026-01-20T00:00:00Z');
const pages = [];
for (let p = 1; p <= 4; p++) pages.push(await (await f(`https://cached.invalid/repos/o/r/pulls/1/comments?per_page=100&page=${{p}}`)).json());
const first = await (await f('https://cached.invalid/repos/o/r/pulls/1/comments?per_page=100')).json();
console.log(JSON.stringify({{ sizes: pages.map(p => p.length), ids: pages.flat().map(c => c.id), first: first.length }}));""")
    assert got["sizes"] == [100, 50, 0, 0] and got["first"] == 100         # no page parameter is page 1
    assert sorted(got["ids"]) == list(range(150))                          # none repeated, none missing
