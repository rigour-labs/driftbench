import os
import json
import click
import litellm
from dotenv import load_dotenv
from runner.engine import BenchmarkEngine, Task

load_dotenv()

SYSTEM_PROMPT = """You are an expert software engineer. 
Your goal is to implement the requested change in the provided codebase.
Follow the project's existing patterns, security standards, and architectural style.
Return your answer ONLY as a git-style unified diff (patch).
Do not include any explanation or markdown formatting around the diff.
"""

USER_PROMPT_TEMPLATE = """
Project Repository: {repository}
Goal (Intent): {intent}

Below is the context of the relevant files for this task. 
Please provide a git patch that implements the intent.

{context}
"""

class LLMHarness:
    def __init__(self, model: str):
        self.model = model
        self.engine = BenchmarkEngine(os.getcwd())

    def get_repo_context(self, repo_path: str) -> str:
        """Simple context collector - in a real scenario, this would use a more advanced RAG or tree-sitter approach."""
        context = []
        # For simplicity in the benchmark, we provide a few top-level files or specific files if we had them.
        # Here we just list the files to give the LLM an idea.
        for root, dirs, files in os.walk(repo_path):
            if '.git' in dirs: dirs.remove('.git')
            if 'node_modules' in dirs: dirs.remove('node_modules')
            if 'venv' in dirs: dirs.remove('venv')
            
            for f in files[:20]: # Limit to avoid token blast
                try:
                    p = os.path.join(root, f)
                    with open(p, 'r', errors='ignore') as src:
                        content = src.read()[:2000]
                        context.append(f"--- File: {os.path.relpath(p, repo_path)} ---\n{content}\n")
                except:
                    continue
        return "\n".join(context)

    def generate_patch(self, task: Task) -> str:
        """Prompts the LLM to generate a patch for the given task."""
        repo_path = self.engine.setup_repo(task.repository, task.base_sha)
        context = self.get_repo_context(repo_path)
        
        prompt = USER_PROMPT_TEMPLATE.format(
            repository=task.repository,
            intent=task.intent,
            context=context
        )
        
        click.echo(f"    🤖 Prompting {self.model}...")
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        
        patch_text = response.choices[0].message.content
        # Basic cleanup if model includes markdown
        if "```diff" in patch_text:
            patch_text = patch_text.split("```diff")[1].split("```")[0]
        elif "```" in patch_text:
            patch_text = patch_text.split("```")[1].split("```")[0]
            
        return patch_text.strip()

    def run_task(self, task: Task):
        click.echo(f"🚀 Benchmarking {self.model} on: {task.id}")
        
        # 1. Generate Patch
        patch_content = self.generate_patch(task)
        
        # 2. Save Patch
        model_slug = self.model.replace("/", "_")
        patch_dir = f"results/patches/{model_slug}/{task.id}"
        os.makedirs(patch_dir, exist_ok=True)
        patch_path = os.path.join(patch_dir, "llm_generated.patch")
        
        with open(patch_path, 'w') as f:
            f.write(patch_content)
            
        # 3. Evaluate Patch
        results_dir = f"results/{model_slug}"
        os.makedirs(results_dir, exist_ok=True)
        
        click.echo(f"    🔬 Evaluating generated patch...")
        report = self.engine.evaluate_patch(task, patch_path, f"llm_{model_slug}", results_dir)
        
        # Save detailed report
        report_path = os.path.join(results_dir, f"{task.id}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        if report["detected"]:
            click.secho("    🔴 DRIFT DETECTED", fg='red', bold=True)
        else:
            click.secho("    🟢 NO DRIFT (PASSED)", fg='green', bold=True)

@click.command()
@click.option('--model', default='claude-3-5-sonnet', help='Model to benchmark (e.g., anthropic/claude-4-5-opus, openai/gpt-5.2-codex)')
@click.option('--task', help='Specific task ID to run')
def main(model, task):
    harness = LLMHarness(model)
    
    if task:
        # Load specific task
        task_file = None
        for f in glob.glob("datasets/**/*.json", recursive=True):
            with open(f, 'r') as t:
                data = json.load(t)
                if data["id"] == task:
                    task_file = f
                    break
        if task_file:
            t = Task.from_json(task_file)
            harness.run_task(t)
    else:
        # Run all tasks (caution: expensive)
        task_files = glob.glob("datasets/**/*.json", recursive=True)
        for f in task_files:
            t = Task.from_json(f)
            harness.run_task(t)

if __name__ == "__main__":
    import glob
    main()
