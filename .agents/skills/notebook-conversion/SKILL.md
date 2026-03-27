---
name: notebook-conversion
description: Utilities for converting Jupyter notebooks to Python (.py) and Markdown (.md) formats.
---

# Notebook Conversion Skill

This skill provides tools and instructions for converting Jupyter Notebooks (.ipynb) into Python scripts and Markdown overviews. This is useful for version control, documentation, and making notebooks searchable in a codebase.

## Prerequisites

- Python 3.x
- `nbformat` library installed (e.g., via `uv pip install nbformat`)

## Usage

### Converting a Single Notebook

To convert a notebook, use the provided conversion script:

```bash
python .agents/skills/notebook-conversion/scripts/convert.py <path_to_notebook.ipynb>
```

### Batch Conversion

To convert all notebooks in a directory recursively:

```bash
python .agents/skills/notebook-conversion/scripts/convert.py <directory_path>
```

## Structure of Converted Files

- **.py**: Contains all code cells. Markdown cells are included as comments prefixed with `# `.
- **.md**: Contains all markdown cells and code cells wrapped in ` ```python ` blocks.

## Scripts

- `scripts/convert.py`: The main conversion script.
