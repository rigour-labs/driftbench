import litellm
import os

def test_model(test_name, model_string, use_prompt=False):
    print(f"\n🔬 Testing {test_name}: {model_string}")
    try:
        kwargs = {
            "model": model_string,
            "temperature": 0
        }
        
        if use_prompt:
            kwargs["prompt"] = "System: You are a coding assistant.\n\nUser: Hello"
        else:
            kwargs["messages"] = [{"role": "user", "content": "Hello"}]
            
        # Direct call, relying on LiteLLM's internal provider logic
        response = litellm.completion(**kwargs)
        
        content = ""
        if hasattr(response.choices[0], 'message'):
            content = response.choices[0].message.content
        else:
            content = response.choices[0].text
            
        print(f"✅ Success! Response: {content}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    print("--- Verifying 'SOLID' Configuration with System Keys ---")
    
    # Test 1: OpenAI future model (legacy completion mode)
    # Using the exact string constructed by harness.py from model_config.json
    test_model("GPT-5.1 Codex", "text-completion-openai/gpt-5.1-codex", use_prompt=True)
    
    # Test 2: Anthropic future model (chat mode)
    test_model("Claude Opus 4.5", "anthropic/claude-opus-4-5", use_prompt=False)
