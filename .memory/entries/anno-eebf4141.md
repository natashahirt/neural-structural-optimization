---
id: anno-eebf4141
type: annotation
project: semantopology_hardfork
parent_id: plan-e9993b76
title: 'TOOLING GOTCHA: python must run OUTSIDE the tool sandbox in this repo; sandboxed
  numpy imports die with SIGFPE'
node_label: 'TOOLING GOTCHA: python must run OUTSIDE the tool s'
tags: gotcha,tooling,sandbox,numpy,openblas,environment
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/tests/test_discretization.py@c97b554
session_id: sess-bfd66d57
created_at: '2026-09-04T00:34:24.609552+00:00'
updated_at: '2026-09-04T00:34:24.609552+00:00'
---
Cost roughly an hour of misdiagnosis during stage 1. Any agent running python here must know this BEFORE it starts debugging.

SYMPTOM: any sandboxed python that imports numpy dies with 'Fatal Python error: Floating point exception' / 'Floating point exception: 8'. The traceback points inside numpy's OWN import-time macOS check:
  numpy/linalg/linalg.py:561 in inv
  numpy/lib/polynomial.py:680 in polyfit
  numpy/__init__.py:386 in _mac_os_check
It fires before a single line of project code executes, and pytest reports it during COLLECTION, so it reads convincingly like a broken environment or a corrupt install.

CAUSE: the Cursor tool sandbox blocks syscalls that OpenBLAS uses for CPU/thread detection. The environment is fine. Confirmed by running the identical command with the sandbox disabled: it works instantly.

FIX: run every python/pytest command with required_permissions: ["all"] and invoke the interpreter directly:
  cd "<repo>" && PYTHONPATH="$PWD" /opt/miniconda3/envs/neural_structural_opt/bin/python ...
Do NOT use `conda run` - it is sandboxed too unless the permission is passed, and it swallows the real traceback behind a conda wrapper error.

WHY IT MISLEADS SO WELL, and the actual lesson: it appeared immediately after a `conda install scikit-sparse`, whose solve genuinely DID remove the numpy-base package record. That gave a coherent, entirely wrong causal story - 'my install broke numpy' - and three rounds of conda surgery followed, all no-ops. numpy in this env is a PIP wheel with its own bundled libopenblas64_ in site-packages/numpy/.dylibs/, so the conda numpy record was never load-bearing and conda could not have fixed it. The tell that was available the whole time and got skipped: run the failing command once with the sandbox off BEFORE forming any hypothesis about the environment. Environment archaeology should come second, not first.

OTHER SANDBOX-SENSITIVE COMMANDS: `git stash`/`git commit` need git_write; conda/pip installs need full_network. Plain `grep`/`ls`/`git log` are fine sandboxed.
