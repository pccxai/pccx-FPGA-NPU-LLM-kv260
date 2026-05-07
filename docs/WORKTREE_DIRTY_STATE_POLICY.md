# Worktree Dirty-State Policy

Use this policy when `git status` shows modified, deleted, staged, or
untracked files that are unrelated to the task currently being worked.

## Rule

Do not stash, reset, delete, or commit unrelated dirty state.

Unrelated dirty state may be another maintainer's local work, generated
evidence that still needs review, or a previous agent handoff. Preserve
it and move the current task to a clean surface instead of hiding it in
a stash or mixing it into the current PR.

## Required response

1. Stop before editing files that overlap the dirty paths.
2. Record the dirty state with:

   ```bash
   git status --short --branch
   git diff --name-only
   git diff --cached --name-only
   ```

3. File a GitHub issue titled `Unrelated dirty state in <worktree-or-branch>`.
   Include:

   - the worktree path or branch,
   - the command that exposed the dirty state,
   - the `git status --short --branch` output,
   - the dirty file list,
   - whether any paths overlap the intended task.

4. Continue the requested task from a clean worktree or a new branch.
5. Link the issue from the PR body if the dirty state affected where the
   task was implemented.

## What not to do

- Do not run `git stash` to clear the tree.
- Do not run `git reset --hard`, `git clean`, or checkout commands that
  discard local state unless the owner explicitly requests it.
- Do not stage or commit unrelated dirty files.
- Do not include unrelated dirty paths in a documentation, RTL, driver,
  or evidence PR just because they are already present.
- Do not paste sensitive local paths, credentials, or private log
  contents into a public issue. Summarize those entries instead.

## If paths overlap

When unrelated dirty state touches a file needed for the current task,
pause and ask for ownership direction. If the owner says to proceed,
make the smallest possible edit that preserves the existing change and
call out the overlap in the PR body.
