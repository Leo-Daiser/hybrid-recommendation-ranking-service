# GEMINI.md

## Antigravity Agent Behavior

Use autonomous phase execution.

Do not pause to ask the user "Should I continue?" after every file or small step.

If a requested phase is clear, complete the entire phase before asking for review.

Ask only for confirmation when an action is destructive, irreversible, outside the active phase, or touches secrets, raw data, remote git, or destructive database operations.

When implementing a phase:
1. Inspect the repository.
2. Create a concise phase plan.
3. Implement files.
4. Add tests.
5. Run tests.
6. Fix failures.
7. Update README.
8. Report final status.

Do not implement future phases early.

Do not rewrite the whole project if a smaller targeted change is enough.

Prefer editing existing code over duplicating logic.

Never hide failing tests. If tests fail and cannot be fixed within the current scope, report the exact failure and the minimal fix needed.
