---
name: notebook-management
description: Utilities for managing Jupyter Notebooks and their converted counterparts in a repository.
---

# Notebook Management Skill

This skill provides utilities for health checks, cleanup, and status reporting for Jupyter Notebooks and their generated `.py` and `.md` versions.

## Usage

### Cleanup Generated Files

To remove all `.py` and `.md` files that have been generated from notebooks:

```bash
python .agents/skills/notebook-management/scripts/manage.py --action clean <directory_path>
```

### Conversion Status Check

To list notebooks and their corresponding conversion status:

```bash
python .agents/skills/notebook-management/scripts/manage.py --action status <directory_path>
```

## Scripts

- `scripts/manage.py`: The main management utility.
