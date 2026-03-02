# RLAIF Pipeline Setup Guide

## 1. HuggingFace Setup

### Step 1: Create a HuggingFace account
Go to https://huggingface.co/join and sign up (or log in).

### Step 2: Create the dataset repo
Go to https://huggingface.co/new-dataset and create:
- **Owner**: `rigour-labs` (your organization)
- **Dataset name**: `rigour-rlaif-data`
- **License**: Apache 2.0
- **Visibility**: Public

This creates `rigour-labs/rigour-rlaif-data` where the pipeline uploads training data.

### Step 3: Generate an API token
Go to https://huggingface.co/settings/tokens and click **New token**:
- **Name**: `rigour-rlaif-pipeline`
- **Type**: `Write` (needs write access to upload files)
- Copy the token (starts with `hf_...`)

You will use this token as `HF_TOKEN` in GitHub Actions secrets (Step 5).

## 2. GitHub Actions Secrets Setup

### Step 4: Navigate to repo settings
Go to your driftbench GitHub repo:
`https://github.com/rigour-labs/driftbench/settings/secrets/actions`

Or: **Repo → Settings → Secrets and variables → Actions → New repository secret**

### Step 5: Add required secrets

Add these secrets one by one (click **New repository secret** for each):

| Secret Name | Value | Required? |
|---|---|---|
| `HF_TOKEN` | Your HuggingFace write token (`hf_...`) | Yes (for upload) |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (`sk-ant-...`) | Yes (default provider) |

### Step 6: Add optional provider secrets (if using other providers)

Only add the ones you plan to use:

| Secret Name | Value | When needed |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` | Using OpenAI as teacher |
| `DEEPSEEK_API_KEY` | `sk-...` | Using DeepSeek as teacher |
| `GROQ_API_KEY` | `gsk_...` | Using Groq as teacher |
| `TOGETHERAI_API_KEY` | Together AI key | Using Together as teacher |
| `FIREWORKS_API_KEY` | Fireworks AI key | Using Fireworks as teacher |
| `MISTRAL_API_KEY` | Mistral key | Using Mistral as teacher |
| `GEMINI_API_KEY` | Google AI key | Using Gemini as teacher |

## 3. Running the Pipeline

### Manual trigger (recommended first time)
Go to **Actions → RLAIF Training Data Pipeline → Run workflow**:
- **repos**: Start with a single repo like `expressjs/express`
- **model_provider**: `anthropic` (or whichever you set up)
- **model_name**: `claude-sonnet-4-20250514`
- Click **Run workflow**

### Local test run
```bash
cd driftbench

# Install dependencies
pip install -r requirements.txt
pip install litellm huggingface_hub

# Test with one repo
ANTHROPIC_API_KEY=sk-ant-... python -m rlaif.generate \
  --repo "expressjs/express" \
  --output rlaif/data

# Format DPO pairs
python -m rlaif.format_dpo \
  --db rlaif/data/training_data.db \
  --output rlaif/data
```

### Automatic schedule
The pipeline runs automatically every Sunday at 11 PM IST (5:30 PM UTC).
Edit `.github/workflows/rlaif-pipeline.yml` to change the schedule.

## 4. Verify It Works

After a successful run, check:
1. **GitHub Actions artifacts**: Download `rlaif-training-data-*` artifact
2. **HuggingFace**: Visit `https://huggingface.co/datasets/rigour-labs/rigour-rlaif-data`
3. **Files uploaded**: `sft_data.jsonl`, `dpo_data.jsonl`, `category_stats.json`
