from bench.harness.diffstat import changed_lines, parse_hunks

DIFF = """diff --git a/app.py b/app.py
index 1..2 100644
--- a/app.py
+++ b/app.py
@@ -1,3 +1,4 @@
 def f():
-    return 1
+    return 2
+    # done
 x = 1
@@ -20,2 +21,1 @@
--- removed sql comment line
 keep
diff --git a/gone.sql b/gone.sql
deleted file mode 100644
--- a/gone.sql
+++ /dev/null
@@ -1,2 +0,0 @@
-select 1;
-select 2;
diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,1 @@
+print("hi")
"""


def test_hunks_paths_and_first_added_line():
    hunks = parse_hunks(DIFF)
    assert [(h.path, h.first_line, h.added, h.removed) for h in hunks] == [
        ("app.py", 2, 2, 1),
        ("app.py", 21, 0, 1),      # a removed "-- comment" line is content, not a header
        ("gone.sql", 0, 0, 2),     # deleted file kept under its old path
        ("new.py", 1, 1, 0),
    ]
    assert changed_lines(hunks) == 7
    assert [h.last_line for h in hunks] == [3, 21, 0, 1]


def test_plain_unified_diff_without_git_header():
    hunks = parse_hunks("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n")
    assert [(h.path, h.first_line) for h in hunks] == [("x.py", 1)]
