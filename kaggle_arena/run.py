"""Pre-emption on a free Kaggle GPU: which model size raises what reviewers flagged.

Runs as a Kaggle script kernel (GPU T4, internet on). For each GGUF model in
the ladder it starts a llama.cpp OpenAI-compatible server on the GPU, points
Rigour at it (the same pipeline as --max and bring-your-own-key: reference
pack, two passes, self-check), and runs the arena's pre-emption sample. Each
model's results, with Rigour's funnel (proposed -> withdrawn -> kept), go to
/kaggle/working/results/<model>/<repo>/pre-pr/<tool>.json.

The workflow that pushes this kernel replaces PARAMS below; run locally, the
script only prints its plan.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# build.py replaces the placeholder with the run's parameters (JSON).
PARAMS_JSON = r'''__PARAMS__'''
PARAMS = json.loads(PARAMS_JSON) if PARAMS_JSON.startswith("{") else {}
WORK = Path("/kaggle/working")
SRC = Path("/kaggle/tmp") if Path("/kaggle").exists() else Path.cwd() / ".kaggle-local"
NODE_VERSION = "v22.12.0"
PORT = 8000
LLAMA_WHEELS = "https://abetlen.github.io/llama-cpp-python/whl/cu124"


def sh(*cmd: str, cwd: Path | None = None, env: dict | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def setup() -> dict:
    """Node, Rigour built from a ref, driftbench, and the CUDA llama.cpp server. Returns the env for Rigour."""
    SRC.mkdir(parents=True, exist_ok=True)
    node = SRC / f"node-{NODE_VERSION}-linux-x64"
    if not node.exists():
        sh("bash", "-c", f"curl -fsSL https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-x64.tar.xz | tar -xJ -C {SRC}")
    env = {**os.environ, "PATH": f"{node}/bin:{os.environ['PATH']}"}
    rigour = SRC / "rigour"
    sh("git", "clone", "-q", "https://github.com/rigour-labs/rigour.git", str(rigour))
    sh("git", "checkout", "-q", PARAMS.get("rigour_ref", "main"), cwd=rigour)
    # corepack in older Node 22 releases fails on npm's rotated signing keys: install the pinned pnpm with npm.
    pnpm = json.loads((rigour / "package.json").read_text()).get("packageManager", "pnpm@10").split("+")[0]
    sh("npm", "install", "-g", "--silent", pnpm, env=env)
    sh("bash", "-c", "pnpm install --frozen-lockfile --silent && pnpm build", cwd=rigour, env=env)
    driftbench = SRC / "driftbench"
    sh("git", "clone", "-q", "https://github.com/rigour-labs/driftbench.git", str(driftbench))
    sh("git", "checkout", "-q", PARAMS.get("driftbench_ref", "main"), cwd=driftbench)
    sh(sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python[server]", "--extra-index-url", LLAMA_WHEELS)
    env |= {"RIGOUR_CLI": str(rigour / "packages/rigour-cli/dist/cli.js"), "PYTHONPATH": str(driftbench),
            "ARENA_CACHE": str(SRC / "arena"), "ARENA_RIGOUR_TIMEOUT_S": str(PARAMS.get("timeout_s", 1800))}
    return env


def serve(model: dict) -> subprocess.Popen:
    """The model on the GPU behind an OpenAI-compatible API; returns once it answers."""
    server = subprocess.Popen([
        sys.executable, "-m", "llama_cpp.server", "--hf_model_repo_id", model["repo"], "--model", model["file"],
        "--n_gpu_layers", "-1", "--n_ctx", str(model.get("n_ctx", 32768)), "--host", "127.0.0.1", "--port", str(PORT),
        "--model_alias", "local",
    ])
    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
            return server
        except OSError:
            if server.poll() is not None:
                raise RuntimeError(f"llama.cpp server exited while loading {model['name']}")
            time.sleep(10)
    server.kill()
    raise RuntimeError(f"{model['name']} did not come up")


def evaluate(model: dict, env: dict) -> None:
    """Pre-emption for each eval repo with Rigour pointed at the served model."""
    driftbench = SRC / "driftbench"
    out = WORK / "results" / model["name"]
    config = driftbench / "arena" / "configs" / f"kaggle-{model['name']}.json"
    config.write_text(json.dumps({"note": f"{model['name']} on a Kaggle T4 via llama.cpp", "config": "rigour-default.yml", "flags": [
        "--provider", "openai", "--api-base-url", f"http://127.0.0.1:{PORT}/v1", "--model-name", "local", "-k", "local"]}))
    code = (
        "import json, sys; from pathlib import Path\n"
        "from arena import preemption; from arena.corpus import Corpus; from arena.repos import clone\n"
        "from arena.__main__ import _rigour_config\n"
        "repo, n, tool, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])\n"
        "corpus = Corpus.load(Path('arena/corpora') / (repo.replace('/', '__') + '.json'))\n"
        "git = clone(repo)\n"
        "prs = preemption.select(git, corpus.prs, n)\n"
        "results = preemption.run(git, prs, _rigour_config(tool)) | {'repo': repo}\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / (tool + '.json')).write_text(json.dumps(results, indent=1))\n"
        "print(repo, len(results['prs']), 'PRs', flush=True)\n"
    )
    for repo in PARAMS.get("repos", []):
        target = out / repo.replace("/", "__") / "pre-pr"
        started = time.time()
        sh(sys.executable, "-c", code, repo, str(PARAMS.get("per_repo", 5)), f"kaggle-{model['name']}", str(target), cwd=driftbench, env=env)
        print(f"{model['name']} {repo}: {time.time() - started:.0f}s", flush=True)


def main() -> None:
    plan = {k: PARAMS.get(k) for k in ("rigour_ref", "driftbench_ref", "repos", "per_repo", "models")}
    print("plan:", json.dumps(plan, indent=1), flush=True)
    if not Path("/kaggle").exists():
        return  # local: plan only
    env = setup()
    for model in PARAMS.get("models", []):
        server = serve(model)
        try:
            evaluate(model, env)
        except subprocess.CalledProcessError as error:
            print(f"{model['name']} failed: {error}", flush=True)  # the next model still runs
        finally:
            server.terminate()
            server.wait(timeout=60)


if __name__ == "__main__":
    main()
