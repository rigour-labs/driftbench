import os
import json
import glob
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from runner.engine import BenchmarkEngine, Task

app = FastAPI(title="DriftBench Cloud Runner")

# Persistence (Simple JSON files for now)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class RunRequest(BaseModel):
    task_id: str

class LeaderboardEntry(BaseModel):
    task_id: str
    status: str
    score: float
    timestamp: str

@app.get("/")
async def root():
    return {"status": "online", "service": "DriftBench Cloud Runner"}

@app.get("/tasks")
async def list_tasks():
    """Lists all available benchmark tasks."""
    task_files = glob.glob("datasets/**/*.json", recursive=True)
    tasks = []
    for f in task_files:
        try:
            with open(f, 'r') as t:
                data = json.load(t)
                tasks.append({
                    "id": data["id"],
                    "name": data["name"],
                    "category": data["category"],
                    "repository": data["repository"]
                })
        except:
            continue
    return tasks

@app.post("/run/{task_id}")
async def run_task(task_id: str, background_tasks: BackgroundTasks):
    """Triggers a benchmark task by ID."""
    # Find task file
    task_path = None
    for f in glob.glob("datasets/**/*.json", recursive=True):
        with open(f, 'r') as t:
            data = json.load(t)
            if data["id"] == task_id:
                task_path = f
                break
    
    if not task_path:
        raise HTTPException(status_code=404, detail="Task not found")

    background_tasks.add_task(execute_benchmark, task_path)
    return {"status": "queued", "task_id": task_id}

@app.get("/leaderboard")
async def get_leaderboard():
    """Retrieves the aggregated leaderboard results."""
    results = []
    for f in glob.glob(f"{RESULTS_DIR}/*.json"):
        with open(f, 'r') as r:
            results.append(json.load(r))
    return results

def execute_benchmark(task_path: str):
    """Worker function to run the benchmark engine."""
    engine = BenchmarkEngine()
    task = Task.from_json(task_path)
    report = engine.evaluate_task(task)
    
    # Save result
    result_path = os.path.join(RESULTS_DIR, f"{task.id}.json")
    with open(result_path, 'w') as f:
        json.dump(report, f, indent=2)
