import os
import sys
from dotenv import load_dotenv
import litellm

# Load env vars
load_dotenv()

# Check keys
print(f"ANTHROPIC_KEY: {'Found' if os.getenv('ANTHROPIC_API_KEY') else 'MISSING'}")
print(f"OPENAI_KEY:    {'Found' if os.getenv('OPENAI_API_KEY') else 'MISSING'}")
print(f"GEMINI_KEY:    {'Found' if os.getenv('GEMINI_API_KEY') else 'MISSING'}")

models_to_test = [
    "anthropic/claude-opus-4-5",          # Alias from screenshot
    "openai/gpt-5.2-codex",               # Snapshot name from screenshot
    "gemini/gemini-3-pro-preview"         # Model ID from screenshot
]

print("\n--- Testing Model Connectivity ---")
for model in models_to_test:
    print(f"\nTesting {model}...")
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hello. Just say 'OK'."}],
            max_tokens=10
        )
        print(f"✅ Success! Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
