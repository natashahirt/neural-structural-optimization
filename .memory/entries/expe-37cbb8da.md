---
id: expe-37cbb8da
type: experiment
project: semantopology_hardfork
parent_id: plan-e9993b76
title: 'RESOLVED: the CHOLMOD crash is an OpenMP race, not a size limit; OMP_NUM_THREADS=1
  fixes it and is 9x FASTER'
node_label: 'RESOLVED: the CHOLMOD crash is an OpenMP race, not'
tags: cholmod,openmp,segfault,resolved,performance,stage-3,correction
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/__init__.py@738fc8b
session_id: sess-bfd66d57
created_at: '2026-09-04T01:23:29.243325+00:00'
updated_at: '2026-09-04T01:23:29.243325+00:00'
invalidates: expe-75e27433
results: '[{"metric": "cholmod_crash_rate_96x192_baseline", "value": 60, "unit": "%",
  "split": "multistory_building 96x192, 37152 dofs, 5 trials, no OMP pin", "criterion":
  "previously recorded as SAFE; it is not", "source": "stage-3-scoping-rerun"}, {"metric":
  "cholmod_crash_rate_128x256_baseline", "value": 80, "unit": "%", "split": "multistory_building
  128x256, 65920 dofs, 5 trials, no OMP pin", "criterion": "nondeterministic, not
  a hard limit", "source": "stage-3-scoping-rerun"}, {"metric": "cholmod_crash_rate_with_omp_pin",
  "value": 0, "unit": "%", "split": "96x192 and 128x256, 5 trials each, OMP_NUM_THREADS=1",
  "criterion": "must be 0", "source": "stage-3-scoping-rerun"}, {"metric": "solve_time_128x256_supernodal_1thread",
  "value": 307.9, "unit": "ms", "split": "factorize+solve, mean of 3, 65920 dofs,
  OMP_NUM_THREADS=1", "criterion": "vs SuperLU 2763.4ms = 9.0x", "source": "stage-3-scoping-rerun"},
  {"metric": "torch_matmul_with_pin_and_restore", "value": 44.0, "unit": "ms", "split":
  "1500x1500 matmul, OMP_NUM_THREADS=1 + torch.set_num_threads(8)", "criterion": "vs
  79.1ms torch default; faster, not slower", "source": "stage-3-scoping-rerun"}]'
---
Supersedes expe-75e27433, which called this a size threshold at ~50k dofs. That was wrong, and wrong in an instructive way: I bisected with ONE trial per size. The crash is nondeterministic, so a single green run measured luck, not capability.

WHAT IT ACTUALLY IS. Repeating 5 trials per configuration:
  32x64     4,290 dofs   5/5 OK
  64x128   16,576 dofs   5/5 OK
  96x192   37,152 dofs   2/5 OK, 3/5 SEGFAULT   <- I had recorded this size as safe
  128x256  65,920 dofs   1/5 OK, 4/5 SEGFAULT
So 96x192, which the superseded entry declared below the threshold, fails 60% of the time. There is no safe size, only a probability that rises with size.

CAUSE: an OpenMP threading race in CHOLMOD's supernodal factorization, which is why it only appears once matrices are large enough for that path to thread. libcholmod links libomp AND Apple Accelerate while numpy ships its own bundled OpenBLAS, so more than one OpenMP runtime is live in the process.

FIX: OMP_NUM_THREADS=1, set before any native library loads. 5/5 clean at both 96x192 and 128x256, and the answer matches SuperLU exactly (187.7676).

THIS IS NOT A SPEED SACRIFICE - IT IS THE OPPOSITE. Single-threaded supernodal beats the racy multithreaded version outright, so the contention was pure overhead:
  96x192   supernodal 1-thread 137.5ms  |  multithreaded was 171.9ms  |  simplicial 622.1ms  |  SuperLU 836.6ms
  128x256  supernodal 1-thread 307.9ms  |  multithreaded SEGFAULTS    |  simplicial 2061.8ms |  SuperLU 2763.4ms
At the golden run's 128x256 that is 9.0x faster than SuperLU and 6.7x faster than simplicial - a 5.6x gain over the best previously-working option, and it lands most of the physics half of the stage 2 speed goal for free.

TORCH IS NOT COLLATERAL DAMAGE. OMP_NUM_THREADS=1 would leave torch single-threaded (1500x1500 matmul 139.4ms vs 79.1ms default), but torch.set_num_threads() overrides it completely and setting it explicitly is FASTER than torch's own default: 44.0ms. So neural_structural_optimization/__init__.py pins OMP_NUM_THREADS=1 at the top, before numpy/scipy/sksparse can load, and exposes configure_torch_threads() to restore the torch pool afterwards. Net: stable CHOLMOD, 9x faster physics, and faster torch.

RELATED HAZARD WORTH REMEMBERING: CHOLMOD cannot be caught and retried. Provoking CholmodNotPositiveDefiniteError corrupts its state so the NEXT solve in the process segfaults. That ruled out a try/except fallback, and it is why the right-wall compliance comparison is a comment in test_discretization.py rather than an assertion - the test would have taken the rest of the suite down with it.
