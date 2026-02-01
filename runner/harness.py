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

CRITICAL INSTRUCTIONS:
1. Return your answer ONLY as a standard git unified diff (patch).
2. Start the response immediately with '--- ' or '+++ '.
3. Do NOT include any conversational text, markdown formatting (no ```diff), or explanations.
4. Ensure the context lines correctly match the provided source.
5. Use standard headers: '--- a/path/to/file' and '+++ b/path/to/file'.
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
        """Collects repo context with a strict budget to avoid token overflow."""
        context = []
        total_chars = 0
        MAX_CONTEXT_CHARS = 400000  # ~100k tokens budget
        
        for root, dirs, files in os.walk(repo_path):
            if total_chars >= MAX_CONTEXT_CHARS:
                break
                
            # Ignore binary and hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__', 'dist', 'build']]
            
            for f in files:
                if total_chars >= MAX_CONTEXT_CHARS:
                    break
                    
                # Skip common non-code/heavy files
                if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf', '.zip', '.lock', '-lock.json')):
                    continue
                    
                try:
                    p = os.path.join(root, f)
                    with open(p, 'r', errors='ignore') as src:
                        # Use a smaller slice per file for massive repos
                        content = src.read()[:1500] 
                        entry = f"--- File: {os.path.relpath(p, repo_path)} ---\n{content}\n"
                        context.append(entry)
                        total_chars += len(entry)
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
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0 # Force deterministic and cleaner output
            )
            
            patch_text = response.choices[0].message.content
            
            # Robust extraction using Regex
            import re
            
            # Strategy 1: Look for a markdown code block tagged with 'diff'
            # This handles: ```diff\n--- a/file\n+++ b/file\n...```
            markdown_match = re.search(r"```(?:diff)?\s*([\s\S]*?)\s*```", patch_text)
            if markdown_match:
                potential_patch = markdown_match.group(1).strip()
                # sanity check: does it look like a patch?
                if "--- " in potential_patch and "+++ " in potential_patch:
                    return potential_patch

            # Strategy 2: If no markdown block, look for the first occurrence of standard diff headers
            # This handles raw text output: "Here is the patch:\n--- a/file\n+++ b/file..."
            # We look for a line starting with `--- ` followed eventually by `+++ `
            header_match = re.search(r"(--- a/.*[\r\n]+(?:\+\+\+ b/.*))", patch_text, re.MULTILINE)
            if header_match:
                # We found the start. We assume the rest of the text is the patch.
                start_idx = header_match.start(1)
                return patch_text[start_idx:].strip()
            
            # Fallback: Just try to strip generic leading text (fragile, but last resort)
            lines = patch_text.splitlines()
            start_index = -1
            for i, line in enumerate(lines):
                if line.startswith("--- ") or line.startswith("+++ "):
                    start_index = i
                    break
            
            if start_index != -1:
                return "\n".join(lines[start_index:]).strip()
                
            return patch_text.strip()
        except Exception as e:
            click.secho(f"    ❌ LLM Error: {str(e)}", fg='red')
            raise e

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
