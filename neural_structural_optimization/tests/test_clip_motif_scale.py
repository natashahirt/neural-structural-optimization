# lint as python3
"""Physical-scale CLIP crops: geometry plus a skipped CLIP-backed additive path.

The second CLIP path must not change Venice RandomResizedCrop. Empty
``motif_scale_fracs`` is the frozen 250214 baseline. Geometry tests do not
construct CLIPLoss.
"""

import dataclasses
import importlib.util
import os
import types
from pathlib import Path

import numpy as np
import torch
from absl.testing import absltest

from neural_structural_optimization.experiment import (
    GOLDEN,
    physical_motif_scale_fracs,
    venice_250214,
    venice_250214_motif_scale,
)
from neural_structural_optimization.models.loss_clip import physical_scale_boxes

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
        self.assertTrue(golden.neutral_init)
        self.assertEqual(golden.prompt, GOLDEN.prompt)
        self.assertEqual(golden.num_augs, GOLDEN.num_augs)
        self.assertEqual(golden.clip_resize_short_side, GOLDEN.clip_resize_short_side)


class MotifScalePromptOverrideTest(absltest.TestCase):

    def test_slug_is_filesystem_safe(self):
        golden_script = _load_golden_script()
        self.assertEqual(
            golden_script.prompt_slug('butterfly wing venation'),
            'butterfly_wing_venation')

    def test_empty_prompt_is_rejected(self):
        golden_script = _load_golden_script()
        with self.assertRaises(ValueError):
            golden_script.prompt_slug('   ')
        with self.assertRaises(ValueError):
            golden_script.motif_scale_run_config('   ')

    def test_skeletons_keeps_the_existing_results_dir(self):
        golden_script = _load_golden_script()
        path = golden_script.motif_scale_results_dir('skeletons')
        self.assertEqual(path.name, 'clip_motif_scale_neutral_init')

    def test_other_prompt_gets_a_sibling_dir(self):
        golden_script = _load_golden_script()
        path = golden_script.motif_scale_results_dir('butterfly wing venation')
        self.assertEqual(
            path.name,
            'clip_motif_scale_neutral_init_butterfly_wing_venation')

    def test_prompt_override_does_not_change_the_preset_default(self):
        golden_script = _load_golden_script()
        self.assertEqual(
            venice_250214_motif_scale().to_venice_golden().prompt,
            GOLDEN.prompt)
        overridden = golden_script.motif_scale_run_config(
            'butterfly wing venation')
        self.assertEqual(overridden.prompt, 'butterfly wing venation')
        self.assertTrue(overridden.neutral_init)
        self.assertEqual(
            overridden.motif_scale_fracs,
            physical_motif_scale_fracs(GOLDEN.height, GOLDEN.interval))
        self.assertEqual(
            venice_250214_motif_scale().to_venice_golden().prompt,
            GOLDEN.prompt)


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

    def test_forward_still_takes_raw_image_only(self):
        import inspect
        from neural_structural_optimization.models.loss_clip import CLIPLoss

        params = inspect.signature(CLIPLoss.forward).parameters
        self.assertEqual(list(params), ['self', 'image'])
        self.assertNotIn('motif_image', params)


class ClipDreamLayoutDirTest(absltest.TestCase):

    def test_results_dir_is_a_sibling_of_the_pixel_motif_dirs(self):
        spec = importlib.util.spec_from_file_location(
            'clip_dream_layout',
            _REPO_ROOT / 'script' / 'clip_dream_layout.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        golden_script = _load_golden_script()
        path = module.dream_results_dir(
            'butterfly wing venation', golden=golden_script)
        self.assertEqual(
            path.name, 'clip_dream_layout_whole_butterfly_wing_venation')

    def test_whole_building_config_drops_motif_crops_and_load_frames(self):
        spec = importlib.util.spec_from_file_location(
            'clip_dream_layout',
            _REPO_ROOT / 'script' / 'clip_dream_layout.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        golden_script = _load_golden_script()
        cfg = module.whole_building_dream_config(
            'butterfly wing venation', golden_script)
        self.assertEqual(cfg.motif_scale_fracs, ())
        self.assertFalse(cfg.union_load_sites)
        self.assertEqual(cfg.prompt, 'butterfly wing venation')
        self.assertTrue(cfg.neutral_init)

    def test_gate_helper_agrees_with_the_mass_prior_loss(self):
        spec = importlib.util.spec_from_file_location(
            'clip_dream_layout',
            _REPO_ROOT / 'script' / 'clip_dream_layout.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scaffold = torch.zeros(8, 4)
        scaffold[:, :2] = 1.0
        density = torch.zeros(8, 4)
        density[:, 2:] = 1.0
        gate = module.evaluate_tautology_gate(
            scaffold.numpy(), density.numpy())
        self.assertGreaterEqual(gate['spatial_mass_loss'], 0.99)
        self.assertTrue(gate['passed'])


def _load_dream_layout():
    spec = importlib.util.spec_from_file_location(
        'clip_dream_layout',
        _REPO_ROOT / 'script' / 'clip_dream_layout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_default_clip_dream(
    model, *, steps, lr, control_height, control_width,
):
    """Snapshot of the pre-projection dream loop. Must stay bit-identical."""
    import torch.nn.functional as F

    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(model.seed))
    control = torch.full(
        (1, 1, control_height, control_width),
        float(model.env.args['volfrac']),
        device=model.device,
    )
    noise = torch.rand(control.shape, generator=generator)
    control = torch.nn.Parameter(
        control + 0.01 * (2.0 * noise.to(model.device) - 1.0))
    optimizer = torch.optim.Adam([control], lr=lr)
    losses = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = F.interpolate(
            control,
            size=model.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        logits = F.avg_pool2d(logits, kernel_size=3, stride=1, padding=1)
        logits = logits[:, 0]
        loss = model.get_semantic_loss(logits)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses, logits.detach(), control.detach()


class _StubDreamModel:
    def __init__(self, seed=0, shape=(1, 12, 8), volfrac=0.3, clip_fn=None):
        self.seed = seed
        self.device = torch.device('cpu')
        self.shape = shape
        self.env = types.SimpleNamespace(args={'volfrac': volfrac})
        self._clip_fn = clip_fn or (
            lambda logits: (logits - 0.5).square().mean())

    def get_semantic_loss(self, logits):
        return self._clip_fn(logits)


class ClipDreamInLoopProjectionTest(absltest.TestCase):
    """Flag-gated filter-then-project: default path frozen, new path constrained."""

    def test_helpers_are_the_shared_semantic_prior_ones(self):
        from neural_structural_optimization.models.loss_semantic_prior import (
            heaviside_projection,
            projected_density_view,
            sigma_for_min_feature,
        )
        module = _load_dream_layout()
        self.assertIs(module.heaviside_projection, heaviside_projection)
        self.assertIs(module.projected_density_view, projected_density_view)
        self.assertIs(module.sigma_for_min_feature, sigma_for_min_feature)

    def test_default_path_matches_legacy_loop_bit_for_bit(self):
        module = _load_dream_layout()
        kwargs = dict(
            steps=4, lr=0.2, control_height=6, control_width=4)
        model_new = _StubDreamModel(seed=7)
        model_old = _StubDreamModel(seed=7)
        losses_new, field_new, control_new = module.run_clip_dream(
            model_new, **kwargs)
        losses_old, field_old, control_old = _legacy_default_clip_dream(
            model_old, **kwargs)
        self.assertEqual(losses_new, losses_old)
        torch.testing.assert_close(field_new, field_old, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            control_new, control_old, rtol=0.0, atol=0.0)

    def test_default_path_ignores_volume_weight(self):
        module = _load_dream_layout()
        kwargs = dict(
            steps=3, lr=0.2, control_height=6, control_width=4)
        baseline_losses, baseline_field, _ = module.run_clip_dream(
            _StubDreamModel(seed=3), **kwargs)
        weighted_losses, weighted_field, _ = module.run_clip_dream(
            _StubDreamModel(seed=3),
            volume_weight=1.0e9,
            dream_volume=0.9,
            **kwargs)
        self.assertEqual(baseline_losses, weighted_losses)
        torch.testing.assert_close(
            baseline_field, weighted_field, rtol=0.0, atol=0.0)

    def test_volume_term_pins_projected_mean(self):
        module = _load_dream_layout()
        target = 0.55

        def _zero_clip(logits):
            return logits.new_zeros(())

        kwargs = dict(
            steps=60,
            lr=0.2,
            control_height=8,
            control_width=6,
            project_in_loop=True,
            dream_volume=target,
        )
        _, free, _ = module.run_clip_dream(
            _StubDreamModel(seed=1, volfrac=0.3, clip_fn=_zero_clip),
            volume_weight=0.0,
            **kwargs)
        _, pinned, _ = module.run_clip_dream(
            _StubDreamModel(seed=1, volfrac=0.3, clip_fn=_zero_clip),
            volume_weight=module.DEFAULT_VOLUME_WEIGHT,
            **kwargs)
        free_mean = float(free.mean())
        pinned_mean = float(pinned.mean())
        self.assertGreaterEqual(float(pinned.min()), 0.0)
        self.assertLessEqual(float(pinned.max()), 1.0)
        self.assertLess(abs(pinned_mean - target), abs(free_mean - target))
        self.assertAlmostEqual(pinned_mean, target, delta=0.10)

    def test_scaffold_passthrough_does_not_rank_cut(self):
        module = _load_dream_layout()
        field = np.zeros((8, 4), dtype=np.float32)
        field[:, :2] = 1.0
        upsampled, threshold, scaffold = module.extract_dream_scaffold(
            field, height=16, width=8, target_mean=0.75, passthrough=True)
        self.assertIsNone(threshold)
        np.testing.assert_allclose(scaffold, upsampled)
        self.assertAlmostEqual(float(scaffold.mean()), 0.5, delta=0.02)
        _, cut_threshold, cut = module.extract_dream_scaffold(
            field, height=16, width=8, target_mean=0.75, passthrough=False)
        self.assertIsNotNone(cut_threshold)
        self.assertAlmostEqual(float(cut.mean()), 0.75, delta=0.03)
        self.assertNotAlmostEqual(float(scaffold.mean()), float(cut.mean()), places=2)

    def test_soft_scaffold_preserves_continuous_rank_order(self):
        module = _load_dream_layout()
        field = np.arange(32, dtype=np.float32).reshape(8, 4)
        upsampled, threshold, scaffold = module.extract_dream_scaffold(
            field,
            height=16,
            width=8,
            target_mean=0.75,
            soft_rank=True,
        )
        self.assertIsNone(threshold)
        self.assertEqual(scaffold.shape, (16, 8))
        self.assertGreaterEqual(float(scaffold.min()), 0.0)
        self.assertLessEqual(float(scaffold.max()), 1.0)
        self.assertGreater(len(np.unique(scaffold)), 2)
        order = np.argsort(upsampled, axis=None, kind='stable')
        ranked_in_dream_order = scaffold.reshape(-1)[order]
        self.assertTrue(np.all(np.diff(ranked_in_dream_order) >= 0.0))

    def test_soft_scaffold_rejects_binary_passthrough(self):
        module = _load_dream_layout()
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            module.extract_dream_scaffold(
                np.zeros((8, 4), dtype=np.float32),
                height=16,
                width=8,
                passthrough=True,
                soft_rank=True,
            )

    def test_target_allowed_mean_with_project_in_loop_is_an_error(self):
        module = _load_dream_layout()
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            module.main([
                '--project-in-loop',
                '--target-allowed-mean', '0.6',
                '--dream-only',
            ])

    def test_beta_anneal_scales_with_step_count(self):
        module = _load_dream_layout()
        self.assertAlmostEqual(
            module._annealed_projection_beta(0, 300, 8.0), 1.0)
        self.assertAlmostEqual(
            module._annealed_projection_beta(299, 300, 8.0), 8.0)
        self.assertAlmostEqual(
            module._annealed_projection_beta(5, 11, 8.0), 4.5)


if __name__ == '__main__':
    absltest.main()
