# Prompt Management

This document describes how to manage LLM prompts for the nomad-spec tool using LangFuse and local fallback files.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Local Files    │────►│    LangFuse     │────►│    Runtime      │
│  prompts/*.json │     │  (versioned)    │     │  (PromptManager)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
   version control         labels/versions        fetches prompts
```

**Source of Truth Strategy:**

| Environment | Primary Source | Fallback |
|-------------|----------------|----------|
| Production  | LangFuse (label: `production`) | Local JSON |
| Development | LangFuse (label: `development`) | Local JSON |
| CI/Testing  | Local JSON files | Hardcoded in code |

## CLI Tool: nomad-spec-prompt

A separate CLI tool for managing prompts. This is intentionally separate from `nomad-spec` since it manages the tool itself rather than Nomad job specifications.

### Commands

```bash
# List local prompts
nomad-spec-prompt list

# Push all prompts to LangFuse
nomad-spec-prompt push

# Push specific prompt
nomad-spec-prompt push --name analysis

# Push with specific label
nomad-spec-prompt push --label production

# Pull all prompts from LangFuse
nomad-spec-prompt pull

# Pull specific prompt with version
nomad-spec-prompt pull --name analysis --version 2
```

### Environment Variables

Required when pushing/pulling:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # Optional, defaults to cloud
LANGFUSE_PROMPT_LABEL=development              # Default label for fetching
```

## Local Prompt Files

Prompts are stored in `prompts/` directory as JSON files in LangFuse export format:

```
prompts/
├── analysis.json      # Codebase analysis prompt
├── generation.json    # HCL generation prompt
└── fix.json           # Error fix prompt
```

### File Format

```json
{
  "name": "analysis",
  "type": "chat",
  "prompt": [
    {
      "role": "system",
      "content": "You are a DevOps engineer..."
    }
  ],
  "config": {
    "model": "auto",
    "temperature": 0.1
  },
  "version": 1,
  "labels": ["development"],
  "tags": ["nomad", "analysis"]
}
```

### Current Prompts

| Name | Purpose | Variables |
|------|---------|-----------|
| `analysis` | Analyzes codebase for Dockerfile, dependencies, ports, resources | None |
| `generation` | Generates Nomad HCL job specification | `{error}`, `{current_spec}`, `{memories}` (in fix mode) |
| `fix` | Fixes failed job specifications | `{error}`, `{current_spec}`, `{memories}` |

## Workflows

### Initial Setup (Bootstrap)

```bash
# 1. Configure LangFuse credentials in .env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# 2. Push all prompts to LangFuse
nomad-spec-prompt push --label development
```

### Development Workflow

```bash
# 1. Edit local prompt file
vim prompts/analysis.json

# 2. Push to LangFuse (creates new version)
nomad-spec-prompt push --name analysis
# Output: ✓ analysis → v2 (development)

# 3. Test changes with nomad-spec
nomad-spec generate --path ./my-app

# 4. When satisfied, promote to production
nomad-spec-prompt push --name analysis --label production
```

### Syncing from LangFuse

If prompts are edited directly in LangFuse UI:

```bash
# Pull all prompts to local files
nomad-spec-prompt pull --label development

# Pull specific version
nomad-spec-prompt pull --name analysis --version 3
```

## Programmatic Access

```python
from src.prompts import get_prompt_manager, PromptNotFoundError

manager = get_prompt_manager()

# Get prompt as LangChain ChatPromptTemplate
template = manager.get_prompt("analysis")

# Get raw text (system message content)
text = manager.get_prompt_text("generation")

# Push to LangFuse
result = manager.push_prompt("analysis", label="production")
print(f"Pushed v{result['version']}")

# Pull from LangFuse
path = manager.pull_prompt("analysis", version=2)
```

## Labels and Versioning

LangFuse supports both versions (immutable integers) and labels (mutable pointers):

- **Versions**: Auto-incrementing (v1, v2, v3...) - each push creates a new version
- **Labels**: Named pointers to versions (development, staging, production)

Recommended workflow:
1. Push new changes → creates new version with `development` label
2. Test in development
3. Push same content with `production` label → points production to tested version

## Fallback Behavior

When LangFuse is unavailable or a prompt is missing:

1. Try LangFuse with configured label
2. Fall back to local `prompts/{name}.json`
3. Fall back to hardcoded prompt in code (last resort)

This ensures the tool works offline or without LangFuse configured.

## Future: DSPy Integration

The architecture supports future DSPy optimization:

```
1. COLLECT: LangFuse traces → training dataset
2. OPTIMIZE: DSPy optimizer refines prompts
3. PUBLISH: Save optimized prompt to LangFuse (new version)
4. PROMOTE: Apply "production" label after validation
```

The JSON format includes a reserved `dspy_metadata` field for tracking optimization lineage.
