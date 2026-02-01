import os
import json
import glob
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
from runner.engine import BenchmarkEngine, Task
from runner.harness import LLMHarness

app = FastAPI(title="DriftBench Cloud Runner")
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
    """Aggregates results for the dashboard UI."""
    stats = []
    # Official model list aligned with Anthropic/OpenAI snapshots
    models = ["anthropic/claude-opus-4-5", "openai/gpt-5.2-codex", "anthropic/claude-sonnet-4-5", "openai/gpt-4o"]
    
    for model in models:
        slug = model.replace("/", "_")
        model_results = glob.glob(f"{RESULTS_DIR}/{slug}/*.json")
        
        detected = 0
        total = 0
        for f in model_results:
            with open(f, 'r') as r:
                data = json.load(r)
                if isinstance(data, dict):
                    total += 1
                    if data.get("detected"):
                        detected += 1
        
        ddr = round((detected / total * 100), 1) if total > 0 else 0
        status = "Completed" if total >= 50 else "In Progress" if total > 0 else "Pending"
        
        stats.append({
            "name": model,
            "ddr": ddr,
            "fpr": 0.0,
            "tasks_run": total,
            "status": status
        })
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
        except:
            continue

def execute_benchmark(task_path: str):
    """Worker function to run the benchmark engine."""
    engine = BenchmarkEngine()
    task = Task.from_json(task_path)
    report = engine.evaluate_task(task)
    
    # Save result
    result_path = os.path.join(RESULTS_DIR, f"{task.id}.json")
    with open(result_path, 'w') as f:
        json.dump(report, f, indent=2)
