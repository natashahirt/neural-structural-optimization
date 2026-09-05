# lint as python3
"""Physical-scale CLIP crops: geometry plus a skipped CLIP-backed additive path.

The second CLIP path must not change Venice RandomResizedCrop. Empty
``motif_scale_fracs`` is the frozen 250214 baseline. Geometry tests do not
construct CLIPLoss.
"""

import dataclasses
import importlib.util
import os
from pathlib import Path

import torch
from absl.testing import absltest

from neural_structural_optimization.experiment import (
    GOLDEN,
    physical_motif_scale_fracs,
    venice_250214,
    venice_250214_motif_scale,
)
from neural_structural_optimization.models.loss_clip import physical_scale_boxes
from neural_structural_optimization.models.model_base import Model

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_SCRIPT = _REPO_ROOT / 'script' / 'venice_golden_250214.py'


def _load_golden_script():
    spec = importlib.util.spec_from_file_location(
        'venice_golden_250214', _GOLDEN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clip_weights_are_cached(*model_names) -> bool:
    try:
        from clip import clip as clip_module
    except ImportError:
        return False
    cache = Path(os.path.expanduser('~/.cache/clip'))
    for name in model_names:
        url = clip_module._MODELS.get(name)
        if url is None or not (cache / os.path.basename(url)).exists():
            return False
    return True


def _skip_reason_for_clip_run():
    for module_name in ('clip', 'kornia'):
        if importlib.util.find_spec(module_name) is None:
            return f'{module_name} is not installed'
    if not _clip_weights_are_cached(GOLDEN.clip_model_name, GOLDEN.clip_rn_model_name):
        return 'CLIP checkpoints are not cached; refusing to download in a test'
    return None


class PhysicalScaleGeometryTest(absltest.TestCase):

    def test_side_is_a_fraction_of_elevation_height_not_min_side(self):
        # After Venice short-side-512 resize a 128x256 grid is 512x1024, so
        # storey interval/height = 0.25 is 256 px of *height*, not of width.
        boxes = physical_scale_boxes(1024, 512, 0.25, 4)
        self.assertEqual(tuple(boxes.shape), (4, 4))
        sides = boxes[:, 2]
        torch.testing.assert_close(sides, torch.full_like(sides, 256.0))
        torch.testing.assert_close(boxes[:, 2], boxes[:, 3])

    def test_full_building_crop_is_limited_by_the_short_side(self):
        boxes = physical_scale_boxes(1024, 512, 1.0, 1)
        self.assertEqual(float(boxes[0, 2]), 512.0)
        self.assertEqual(float(boxes[0, 3]), 512.0)

    def test_centers_are_deterministic(self):
        a = physical_scale_boxes(256, 128, 0.25, 4)
        b = physical_scale_boxes(256, 128, 0.25, 4)
        torch.testing.assert_close(a, b)

    def test_golden_physical_fracs_are_building_storey_member(self):
        self.assertEqual(
            physical_motif_scale_fracs(GOLDEN.height, GOLDEN.interval),
            (1.0, 0.25, 0.0625))

    def test_invalid_frac_is_rejected(self):
        with self.assertRaises(ValueError):
            physical_scale_boxes(64, 32, 0.0, 1)
        with self.assertRaises(ValueError):
            physical_scale_boxes(64, 32, 1.5, 1)


class MotifScalePresetDoesNotSilentChangeGoldenTest(absltest.TestCase):

    def test_venice_250214_still_has_empty_fracs(self):
        cfg = venice_250214()
        self.assertEqual(cfg.clip.motif_scale_fracs, ())
        self.assertEqual(cfg.to_venice_golden(), GOLDEN)

    def test_motif_scale_preset_installs_physical_fracs(self):
        cfg = venice_250214_motif_scale()
        self.assertEqual(cfg.name, 'venice_250214_motif_scale')
        self.assertEqual(
            cfg.clip.motif_scale_fracs,
            physical_motif_scale_fracs(GOLDEN.height, GOLDEN.interval))
        golden = cfg.to_venice_golden()
        self.assertNotEqual(golden, GOLDEN)
        self.assertEqual(golden.prompt, GOLDEN.prompt)
        self.assertEqual(golden.num_augs, GOLDEN.num_augs)
        self.assertEqual(golden.clip_resize_short_side, GOLDEN.clip_resize_short_side)


class VeniceMotifDensityRoutingTest(absltest.TestCase):
    """Venice keeps raw primary CLIP while motifs see structural density."""

    class _RecordingClip:

        def __init__(self, motif_scale_fracs):
            self.motif_scale_fracs = motif_scale_fracs
            self.venice_path = object()
            self.primary_image = None
            self.motif_image = None

        def __call__(self, image, *, motif_image=None):
            self.primary_image = image
            self.motif_image = motif_image
            return image.new_tensor(1.0)

    class _RoutingModel:

        def __init__(self, clip_loss, physical_density):
            self.clip_loss = clip_loss
            self.physical_density = physical_density
            self.physical_density_calls = 0

        def _clip_sees_raw_design(self):
            return True

        def get_physical_density(self, logits):
            self.physical_density_calls += 1
            return self.physical_density

    def test_motif_scales_receive_physical_density_only(self):
        raw = torch.full((1, 4, 2), 7.0)
        physical = torch.full((1, 4, 2), 0.3)
        clip_loss = self._RecordingClip((0.25, 0.0625))
        model = self._RoutingModel(clip_loss, physical)

        Model.get_semantic_loss(model, raw)

        self.assertIs(clip_loss.primary_image, raw)
        self.assertIs(clip_loss.motif_image, physical)
        self.assertEqual(model.physical_density_calls, 1)

    def test_empty_scales_preserve_raw_only_venice_route(self):
        raw = torch.full((1, 4, 2), 7.0)
        physical = torch.full((1, 4, 2), 0.3)
        clip_loss = self._RecordingClip(())
        model = self._RoutingModel(clip_loss, physical)

        Model.get_semantic_loss(model, raw)

        self.assertIs(clip_loss.primary_image, raw)
        self.assertIsNone(clip_loss.motif_image)
        self.assertEqual(model.physical_density_calls, 0)


class MotifScaleClipPathTest(absltest.TestCase):
    """CLIP-backed: second path is additive; empty fracs match Venice-only."""

    def test_empty_fracs_leave_venice_loss_unchanged_and_record_nothing(self):
        reason = _skip_reason_for_clip_run()
        if reason:
            self.skipTest(reason)
        golden_script = _load_golden_script()

        image = torch.rand(1, 32, 64)
        torch.manual_seed(0)
        venice_only = golden_script.build_clip_loss(GOLDEN)
        torch.manual_seed(0)
        loss_a = venice_only(image)
        self.assertEqual(venice_only.last_motif_scale_losses, {})

        torch.manual_seed(0)
        still_empty = golden_script.build_clip_loss(
            dataclasses.replace(GOLDEN, motif_scale_fracs=()))
        torch.manual_seed(0)
        loss_b = still_empty(image)
        torch.testing.assert_close(loss_a, loss_b)

    def test_physical_fracs_add_a_recorded_second_term(self):
        reason = _skip_reason_for_clip_run()
        if reason:
            self.skipTest(reason)
        golden_script = _load_golden_script()

        fracs = physical_motif_scale_fracs(GOLDEN.height, GOLDEN.interval)
        scaled = dataclasses.replace(
            GOLDEN, motif_scale_fracs=fracs, motif_scale_crops=2,
            motif_scale_weight=1.0)
        image = torch.rand(1, 32, 64)

        torch.manual_seed(0)
        venice_only = golden_script.build_clip_loss(GOLDEN)
        torch.manual_seed(0)
        loss_venice = venice_only(image)

        torch.manual_seed(0)
        both = golden_script.build_clip_loss(scaled)
        torch.manual_seed(0)
        loss_both = both(image)

        recorded = both.last_motif_scale_losses
        self.assertIn('mean', recorded)
        self.assertIn('frac=1.0000', recorded)
        self.assertIn('frac=0.2500', recorded)
        self.assertIn('frac=0.0625', recorded)
        extra = image.new_tensor(recorded['mean'])
        torch.testing.assert_close(
            loss_both, loss_venice + extra, rtol=1e-4, atol=1e-4)
        self.assertGreater(float(loss_both), float(loss_venice))


if __name__ == '__main__':
    absltest.main()
