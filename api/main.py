import os
import json
import glob
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from runner.engine import BenchmarkEngine, Task
from runner.harness import LLMHarness

app = FastAPI(title="DriftBench Cloud Runner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="api/templates")

# Persistence (Simple JSON files for now)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Diagnostic: Check for API keys
print("--- Environment Diagnostic ---")
print(f"ANTHROPIC_API_KEY: {'Found' if os.getenv('ANTHROPIC_API_KEY') else 'NOT FOUND'}")
print(f"OPENAI_API_KEY:    {'Found' if os.getenv('OPENAI_API_KEY') else 'NOT FOUND'}")
print("------------------------------")

class RunRequest(BaseModel):
    task_id: str

class LeaderboardEntry(BaseModel):
    task_id: str
    status: str
    score: float
    timestamp: str

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "driftbench"}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serves the premium leaderboard dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/stats")
async def get_stats():
    """Aggregates results for the dashboard UI dynamically."""
    stats = []
    
    # Discovery: Find all model directories that have json results
    if not os.path.exists(RESULTS_DIR):
        return []
        
    # Get all subdirectories in results
    model_slugs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    for slug in model_slugs:
        # Ignore non-result system directories
        if slug in ["patches", "studio"]:
            continue
            
        model_results = glob.glob(f"{RESULTS_DIR}/{slug}/*.json")
        if not model_results:
            continue
            
        detected = 0
        total = 0
        for f in model_results:
            try:
                with open(f, 'r') as r:
                    data = json.load(r)
                    if isinstance(data, dict):
                        total += 1
                        if data.get("detected"):
                            detected += 1
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"Warning: Skipping malformed result file {f}: {e}")
                continue

        ddr = round((detected / total * 100), 1) if total > 0 else 0
        status = "Completed" if total >= 50 else "In Progress"
        
        # Heuristic to restore the display name (e.g. anthropic_claude... -> anthropic/claude...)
        # In a production system we'd store a metadata.json, but this works for the current naming convention
        display_name = slug.replace("_", "/", 1) if "_" in slug else slug
        
        stats.append({
            "name": display_name,
            "ddr": ddr,
            "fpr": 0.0,
            "tasks_run": total,
            "status": status
        })
        
    # Sort by DDR (descending) so leaders stay at the top
    stats.sort(key=lambda x: x["ddr"], reverse=True)
    return stats

@app.post("/run-all")
async def trigger_full_benchmark(background_tasks: BackgroundTasks, model: str = "claude-3-5-sonnet"):
    """Triggers a full baseline run across all 50 tasks."""
    background_tasks.add_task(run_full_suite, model)
    return {"status": "batch_queued", "model": model}

def run_full_suite(model: str):
    """Worker to run all tasks for a specific model."""
    harness = LLMHarness(model)
    task_files = glob.glob("datasets/**/*.json", recursive=True)
    for f in task_files:
        try:
            task = Task.from_json(f)
            harness.run_task(task)
        except Exception as e:
            print(f"Error running task {f} for model {model}: {e}")
            continue

def execute_benchmark(task_path: str, workspace_root: str = "."):
    """Worker function to run the benchmark engine."""
    engine = BenchmarkEngine(workspace_root=workspace_root)
    task = Task.from_json(task_path)
    report = engine.evaluate_task(task)

    # Save result
    result_path = os.path.join(RESULTS_DIR, f"{task.id}.json")
    with open(result_path, 'w') as f:
        json.dump(report, f, indent=2)
