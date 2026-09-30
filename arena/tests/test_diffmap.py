from arena.diffmap import Hunk, map_line, parse_hunks, touches

DIFF = """diff --git a/a.ts b/a.ts
--- a/a.ts
+++ b/a.ts
@@ -2,0 +3,2 @@ export const a = 1;
+const inserted1 = 1;
+const inserted2 = 2;
@@ -10 +12 @@ function f() {
-  return x;
+  return y;
@@ -20,3 +22,0 @@
-  gone1();
-  gone2();
-  gone3();
"""


def test_parses_counts_including_implicit_one_and_zero():
    assert parse_hunks(DIFF) == [Hunk(2, 0, 3, 2), Hunk(10, 1, 12, 1), Hunk(20, 3, 22, 0)]


def test_maps_unchanged_lines_by_the_net_shift_of_earlier_hunks():
    hunks = parse_hunks(DIFF)
    assert map_line(hunks, 1) == (1, False)
    assert map_line(hunks, 2) == (2, False)   # insertion is *after* old line 2
    assert map_line(hunks, 3) == (5, False)   # +2 inserted above
    assert map_line(hunks, 15) == (17, False)
    assert map_line(hunks, 30) == (29, False)  # +2, then -3


def test_maps_changed_lines_to_the_replacing_hunk():
    hunks = parse_hunks(DIFF)
    assert map_line(hunks, 10) == (12, True)
    assert map_line(hunks, 21) == (22, True)  # deleted: clamps to where the text was


def test_touches_only_replaced_or_removed_old_lines():
    hunks = parse_hunks(DIFF)
    assert touches(hunks, 10, 10)
    assert touches(hunks, 18, 20)
    assert not touches(hunks, 2, 2)  # a pure insertion touches no old line
    assert not touches(hunks, 11, 19)
