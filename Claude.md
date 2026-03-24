# Claude Workspace Rules

These rules apply to future agent work in this repository unless the user gives a more specific instruction.

## Git Workflow

- Work in the repository at `https://github.com/MelodiousAI/Melodious`.
- When a meaningful unit of work is finished, commit it and push it to GitHub.
- When a code change should be committed, use a related feature branch.
- If no related branch exists, create one before committing.
- Do not commit directly to the main branch unless the user explicitly requests it.
- Do not push `Ahmad_Instructions.md` or `Ahmad_Documentation.md`.

## Code Quality

- Keep code clean, explicit, and well-organized.
- Add clear comments and docstrings for public functions, integration points, and non-obvious logic.
- Prefer simple interfaces and predictable data contracts over clever abstractions.
- Keep detector, graph, export, and evaluation responsibilities separated by module.

## Documentation Discipline

- Update `Ahmad_Instructions.md` whenever the current phase or next steps change.
- Update `Ahmad_Documentation.md` for every meaningful experiment, whether it succeeded or failed.
- Before starting new work, check the relevant sections in `Ahmad_Documentation.md` to avoid repeating failed ideas.
- Document comparisons between approaches, hyperparameters, metrics, and tradeoffs.

## Analysis Expectations

- Compare multiple approaches whenever practical.
- Record both measured metrics and realistic projections for approaches that were scoped but not fully trained.
- Keep enough evidence in the documentation for the final report to explain why one approach was chosen over another.

## Collaboration Contract

- Preserve the detector output contract used by the graph/GNN side.
- Treat integration files and sample outputs as stable interfaces once shared with teammates.
- If a contract changes, update code, docs, and sample outputs together.