# Multi-Agent Drift Category

This directory contains benchmark tasks for evaluating AI agent behavior in multi-agent coordination scenarios.

## Category: Agent Team Drift

This drift category evaluates whether AI agents can successfully coordinate when working in teams, as enabled by frontier models like Opus 4.6 (agent teams) and GPT-5.3-Codex (coworking mode).

## Example Failures

| Failure Type | Description |
|:---|:---|
| Cross-Agent Pattern Conflict | Two agents implement conflicting patterns in the same codebase |
| Handoff Context Loss | Agent B fails to use context properly from Agent A |
| Task Overlap | Multiple agents work on the same file without coordination |
| Checkpoint Drift | Long-running agent degrades in quality over extended execution |

## Task Structure

```
multi-agent/
├── cross-agent-conflict-001/
│   ├── task.json
│   ├── agent_a_prompt.txt
│   ├── agent_b_prompt.txt
│   └── rigour_config.yaml
├── handoff-context-001/
│   └── ...
└── checkpoint-drift-001/
    └── ...
```

## Rigour Configuration

These tasks require Rigour v2.14+ with agent team governance enabled:

```yaml
gates:
  agent_team:
    max_concurrent_agents: 3
    cross_agent_pattern_check: true
    handoff_verification: true
```
