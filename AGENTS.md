# Working with Notebooks

- The `.ipynb` is the source of truth. NEVER read or edit it directly.
- The `.py` (Jupytext percent-format mirror) is what you read and edit.
- Cell indices passed to `nb run` and `nb outputs` are 0-based, counting ALL cells (markdown and code), matching the `# %%` markers in the `.py` file.

## Workflow

Use the `bash` tool to call `./bin/nb`, a bash script available locally.

- ALWAYS pull before reading or editing a notebook:
```
bin/nb pull /abs/path/to/notebook.ipynb
```

- ALWAYS push after editing the .py mirror:
```
bin/nb push /abs/path/to/notebook.ipynb
```

- ONLY run a notebook if specifically asked by the user:
```
bin/nb run /abs/path/to/notebook.ipynb [first_cell [last_cell]]
```

- ASK the user if the notebook has been recently run before reading outputs:
```
bin/nb output /abs/path/to/notebook.ipynb [first_cell [last_cell]]
```

# Style

- NEVER use `map`/`filter`: use comprehensions.
- USE comprehensions for simple transformations; loops for complex ones.
- USE `lambda` only for short inline functions (e.g., sort keys). Named functions otherwise.
- USE built-in generics: `list[int]`, `dict[str, int]`, never `List`/`Dict` from `typing`.
- USE `X | Y` over `Union[X, Y]`; `X | None` over `Optional[X]`.
- USE `pathlib` over `os` for file system operations.
- USE f-strings over `str.format`.
- ONLY use ascii characters.
- NEVER call `tight_layout()` for plots.

# Design

- Favor data-driven solutions: use dicts, mappings, lookup tables over long if/elif chains.
- AVOID deep nesting: prefer early returns, guard clauses, or `match`.
- AVOID nested loops: consider `itertools` or restructuring data.
- NEVER swallow exceptions silently.
- AVOID comments. At most, explain why, never what.
