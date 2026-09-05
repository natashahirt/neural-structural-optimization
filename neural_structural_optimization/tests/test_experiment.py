# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage 5: typed ExperimentConfig, presets, JSON, overrides, print-config.

These tests must not construct CLIPLoss. The venice_250214 preset is the gate:
it round-trips to VeniceGoldenConfig.GOLDEN, which is the same knobs the
golden script executes.
"""

# pylint: disable=missing-docstring

import json
import subprocess
import sys
from pathlib import Path

from absl.testing import absltest

from neural_structural_optimization.experiment import (
    GOLDEN,
    PHYSICS_PENAL,
    SMOKE,
    ExperimentConfig,
    cli_main,
    preset,
    venice_250214,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON = sys.executable


class VenicePresetMatchesGoldenTest(absltest.TestCase):
  """The Stage 5 gate: venice_250214 is GOLDEN, not a restatement of it."""

  def test_from_venice_golden_round_trips_to_golden(self):
    self.assertEqual(
        ExperimentConfig.from_venice_golden(GOLDEN).to_venice_golden(),
        GOLDEN)

  def test_venice_250214_preset_is_golden(self):
    self.assertEqual(venice_250214().to_venice_golden(), GOLDEN)
    self.assertEqual(preset('venice_250214').to_venice_golden(), GOLDEN)

  def test_smoke_preset_is_smoke(self):
    self.assertEqual(
        preset('venice_250214_smoke').to_venice_golden(), SMOKE)

  def test_structural_params_match_the_golden_builder(self):
    params = venice_250214().structural_params()
    self.assertEqual(params.problem_name, GOLDEN.problem_name)
    self.assertEqual(params.width, GOLDEN.width)
    self.assertEqual(params.height, GOLDEN.height)
    self.assertEqual(params.density, GOLDEN.density)
    self.assertEqual(params.interval, GOLDEN.interval)
    self.assertEqual(params.filter_width, GOLDEN.filter_width)

  def test_resolve_names_the_coarse_start_and_physics_defaults(self):
    effective = venice_250214().resolve()['effective']
    self.assertEqual(effective['coarse_width'], 32)
    self.assertEqual(effective['coarse_height'], 64)
    self.assertEqual(effective['coarse_interval'], 16)
    self.assertEqual(effective['schedule_divisor'], 4)
    self.assertEqual(effective['filter_width'], 2.0)
    self.assertFalse(effective['heavyside'])
    self.assertEqual(effective['penal'], PHYSICS_PENAL)
    self.assertEqual(effective['eta'], 0.5)


class ExperimentConfigContractsTest(absltest.TestCase):

  def test_json_round_trip_is_equal(self):
    cfg = venice_250214()
    restored = ExperimentConfig.from_json(cfg.to_json())
    self.assertEqual(restored, cfg)
    self.assertEqual(restored.to_venice_golden(), GOLDEN)

  def test_with_overrides_does_not_mutate_the_original(self):
    cfg = venice_250214()
    before = cfg.to_dict()
    changed = cfg.with_overrides({'optimizer': {'lr': 0.05}})
    self.assertEqual(cfg.to_dict(), before)
    self.assertEqual(cfg.optimizer.lr, GOLDEN.lr)
    self.assertEqual(changed.optimizer.lr, 0.05)
    self.assertEqual(changed.problem.width, GOLDEN.width)

  def test_dotted_overrides(self):
    cfg = venice_250214().with_overrides(**{'problem.width': 64, 'problem.height': 128})
    self.assertEqual(cfg.problem.width, 64)
    self.assertEqual(cfg.problem.height, 128)
    self.assertEqual(cfg.to_venice_golden().width, 64)

  def test_unknown_field_is_rejected(self):
    with self.assertRaises(ValueError):
      venice_250214().with_overrides(**{'optimizer.not_a_field': 1})
    with self.assertRaises(ValueError):
      ExperimentConfig.from_dict({'name': 'x', 'bogus': 1})

  def test_penal_mismatch_is_rejected(self):
    cfg = venice_250214().with_overrides(**{'problem.penal': 2.0})
    with self.assertRaisesRegex(ValueError, 'penal'):
      cfg.resolve()

  def test_indivisible_schedule_is_rejected(self):
    cfg = venice_250214().with_overrides(**{'problem.width': 66})
    with self.assertRaisesRegex(ValueError, 'divisible'):
      cfg.validate()

  def test_run_refuses_a_non_venice_model(self):
    from neural_structural_optimization import experiment as experiment_mod
    cfg = venice_250214().with_overrides(**{'model.kind': 'cnn'})
    with self.assertRaises(NotImplementedError):
      experiment_mod.run(cfg)

  def test_unknown_preset_lists_known_names(self):
    with self.assertRaisesRegex(KeyError, 'venice_250214'):
      preset('not-a-preset')

  def test_physical_motif_scale_fracs_for_golden_grid(self):
    from neural_structural_optimization.experiment import (
        physical_motif_scale_fracs)
    self.assertEqual(
        physical_motif_scale_fracs(256, 64), (1.0, 0.25, 0.0625))

  def test_motif_scale_preset_is_not_a_silent_golden_change(self):
    from neural_structural_optimization.experiment import (
        physical_motif_scale_fracs, venice_250214_motif_scale)
    cfg = venice_250214_motif_scale()
    self.assertEqual(cfg.name, 'venice_250214_motif_scale')
    self.assertEqual(
        cfg.clip.motif_scale_fracs, physical_motif_scale_fracs(256, 64))
    self.assertEqual(venice_250214().to_venice_golden(), GOLDEN)
    self.assertNotEqual(cfg.to_venice_golden(), GOLDEN)
    self.assertFalse(venice_250214().layout.enabled)
    self.assertFalse(cfg.layout.enabled)

  def test_motif_layout_preset_is_opt_in_and_not_a_user_sketch(self):
    from neural_structural_optimization.experiment import (
        physical_motif_scale_fracs, venice_250214_motif_layout)
    cfg = venice_250214_motif_layout()
    self.assertEqual(cfg.name, 'venice_250214_motif_layout')
    self.assertTrue(cfg.layout.enabled)
    self.assertIsNone(cfg.sketch.path)
    self.assertEqual(cfg.sketch.motif_weight, 0.0)
    self.assertEqual(
        cfg.clip.motif_scale_fracs, physical_motif_scale_fracs(256, 64))
    self.assertEqual(
        cfg.resolved_layout_scale_fracs(),
        physical_motif_scale_fracs(256, 64)[1:])
    self.assertEqual(venice_250214().to_venice_golden(), GOLDEN)
    restored = ExperimentConfig.from_json(cfg.to_json())
    self.assertEqual(restored, cfg)
    cfg.validate()


class PrintConfigDoesNotLoadClipTest(absltest.TestCase):
  """`--print-config` must not import CLIP (clip / kornia).

  torch still loads: `experiment` imports `StructuralParams` through
  `structural.problems`, and `structural/__init__.py` eagerly imports `api`,
  which imports torch. That import is left eager on purpose: loading torch
  *after* a CHOLMOD solve is the OpenMP race this repo pins
  `OMP_NUM_THREADS=1` against.

  A subprocess is the only honest check: this test file itself may already
  have pulled numpy via StructuralParams, and a same-process assertion on
  sys.modules would be contaminated by other tests in the session.
  """

  def test_cli_print_config_subprocess_skips_clip_and_kornia(self):
    probe = r"""
import json, sys
from neural_structural_optimization.experiment import cli_main
code = cli_main(["--print-config", "venice_250214"])
assert code == 0
banned = [name for name in ("clip", "kornia") if name in sys.modules]
print(json.dumps({"banned": banned}))
"""
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(_REPO_ROOT)
    result = subprocess.run(
        [_PYTHON, '-c', probe],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        result.returncode, 0,
        msg=f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}')
    # cli_main prints the config JSON; the probe prints a summary line after.
    lines = [line for line in result.stdout.splitlines() if line.startswith('{')]
    self.assertTrue(lines, msg=result.stdout)
    payload = json.loads(lines[-1])
    self.assertEqual(payload['banned'], [], msg=payload)

  def test_print_config_stdout_is_the_resolved_preset(self):
    from io import StringIO
    from unittest import mock
    buf = StringIO()
    with mock.patch('sys.stdout', buf):
      code = cli_main(['--print-config', 'venice_250214'])
    self.assertEqual(code, 0)
    resolved = json.loads(buf.getvalue())
    self.assertEqual(resolved['name'], 'venice_250214')
    self.assertEqual(resolved['optimizer']['lr'], GOLDEN.lr)
    self.assertEqual(resolved['clip']['prompts'], [GOLDEN.prompt])
    self.assertEqual(resolved['effective']['coarse_width'], 32)

  def test_override_flag_reaches_resolve(self):
    from io import StringIO
    from unittest import mock
    buf = StringIO()
    with mock.patch('sys.stdout', buf):
      code = cli_main([
          '--print-config', 'venice_250214',
          '--override', 'optimizer.lr=0.05',
      ])
    self.assertEqual(code, 0)
    resolved = json.loads(buf.getvalue())
    self.assertEqual(resolved['optimizer']['lr'], 0.05)


class SmokeFlagSelectsCoarsePresetTest(absltest.TestCase):

  def test_smoke_flag_prints_the_smoke_preset(self):
    from io import StringIO
    from unittest import mock
    buf = StringIO()
    with mock.patch('sys.stdout', buf):
      code = cli_main(['--smoke'])
    self.assertEqual(code, 0)
    resolved = json.loads(buf.getvalue())
    self.assertEqual(resolved['name'], 'venice_250214_smoke')
    self.assertEqual(resolved['problem']['width'], 32)
    self.assertEqual(resolved['clip']['num_augs'], 4)


if __name__ == '__main__':
  absltest.main()
