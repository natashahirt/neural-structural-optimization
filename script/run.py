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

import sys
import re
import os
from pathlib import Path
from PIL import Image
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import torch

OUTPUT_DIR = Path("script/test_results_pytorch")

# Enable performance optimizations for modern NVIDIA GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

from neural_structural_optimization.structural import utils as pipeline_utils
from neural_structural_optimization.structural import problems
from neural_structural_optimization import models
from neural_structural_optimization.train import PixelRefineTrainer, LBFGS_Optimizer
from neural_structural_optimization.structural.problems import StructuralParams
from neural_structural_optimization.models.loss_clip import CLIPLoss

def create_filename_suffix(suffix_str: str | None) -> str:
    """Normalize user-provided suffix for safe filenames."""
    if not suffix_str:
        return ""

    suffix = suffix_str.replace(' ', '_').replace('=', '_').replace(',', '_')
    suffix = suffix.replace('"', '').replace("'", '')
    suffix = suffix.replace('/', '_').replace('\\', '_')
    suffix = suffix.replace(':', '_').replace(';', '_')

    suffix = re.sub(r'[<>:"/\\|?*]', '_', suffix)

    print(f"Using filename suffix: {suffix}")
    
    return f"_{suffix}"

def slurm_tag() -> str:
    """Return a SLURM-based identifier like '{7322721}' if available, else ''."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"{{{job_id}}}"
    job_name = os.environ.get("SLURM_JOB_NAME")
    if job_name:
        return f"{{{job_name}}}"
    return ""

def load_initial_image(image_path: str | Path, target_shape: tuple[int, int] | None = None) -> torch.Tensor:
    """Load and preprocess an initial image for model initialization."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Initial image not found: {image_path}")
    
    print(f"Loading initial image from: {image_path}")
    
    with Image.open(image_path) as img:
        if img.mode != 'L':
            img = img.convert('L')
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        if img_array.ndim == 2:
            img_array = img_array[np.newaxis, :, :]
        
        img_tensor = torch.from_numpy(img_array)
        
        if target_shape is not None:
            target_h, target_w = target_shape
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        
        print(f"Initial image shape: {img_tensor.shape}")
        return img_tensor


def ensure_output_dir(path: Path = OUTPUT_DIR) -> None:
    """Create output directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_history(ds_history):
    """Ensure history is a list with a step dimension."""
    if not isinstance(ds_history, (list, np.ndarray)):
        ds_history = [ds_history]

    normalized = []
    for ds in ds_history:
        if "step" not in ds.design.dims:
            ds = ds.expand_dims(step=[0])
        normalized.append(ds)
    return normalized


def save_loss_plot(ds_history, filename_suffix: str) -> Path:
    """Plot cumulative losses across stages."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ds in enumerate(ds_history):
        loss_df = ds.rename({"step": "iteration"}).loss.to_pandas().T
        loss_df.cummin().plot(
            linewidth=2,
            label=f"Stage {i+1}: {ds.sizes['y']}x{ds.sizes['x']}",
            ax=ax,
        )

    ax.set_ylabel("Loss")
    ax.set_xlabel("Optimization Step")
    ax.set_title("Loss Comparison Across Stages")
    ax.grid(True)
    ax.legend(title="Resolution", bbox_to_anchor=(1.05, 1), loc="upper left")
    seaborn.despine()
    plt.tight_layout()

    # Use classic Python string formatting to avoid curly brackets in filename
    plot_path = OUTPUT_DIR / ("optimization_comparison_loss_%s.png" % filename_suffix)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_final_designs_plot(ds_history, params, filename_suffix: str, force_legacy: bool = True) -> Path:
    """Plot final designs for each stage."""
    fig, axes = plt.subplots(1, len(ds_history), figsize=(4 * len(ds_history), 6))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    fig.suptitle(f"Final Designs: {params.problem_name}", fontsize=16)

    problem = problems.PROBLEMS_BY_NAME.get(params.problem_name)

    final_designs = []
    for ds in ds_history:
        final_designs.append(ds.design.isel(step=ds.loss.argmin()))

    for i, (ax, final_design) in enumerate(zip(axes, final_designs)):
        if problem:
            if force_legacy:
                image = pipeline_utils.image_from_design(final_design, problem)
                ax.imshow(1.0 - np.array(image), cmap="gray")
            else:
                try:
                    design_array = pipeline_utils.image_from_design_array(final_design, problem)
                    ax.imshow(1.0 - design_array, cmap="gray")
                except Exception:
                    image = pipeline_utils.image_from_design(final_design, problem)
                    ax.imshow(1.0 - np.array(image), cmap="gray")
        else:
            ax.imshow(1.0 - final_design.values, cmap="gray")

        ax.set_title(f"Stage {i+1}: {ds_history[i].sizes['y']}x{ds_history[i].sizes['x']}")
        ax.axis("off")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / ("final_designs_%s.png" % filename_suffix)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path

def main(suffix_str: str | None = None) -> int:
    """Main function with error handling and progress reporting."""
    print("=" * 60)
    print("Neural Structural Optimization - Multi-Method Comparison")
    print("=" * 60)

    user_suffix = suffix_str if suffix_str is not None else " ".join(sys.argv[1:])
    filename_suffix = f"{create_filename_suffix(user_suffix)}{slurm_tag()}"
    ensure_output_dir()

    try:
        max_iterations = 100
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nStarting optimization on device: {device}")

        # Enable CLIP by toggling this flag to True
        use_clip = True
        force_legacy_rendering = False
        clip_loss = None
        if use_clip:
            clip_loss = CLIPLoss(
                clip_model_name="ViT-B/32",
                clip_rn_model_name="RN50",
                device=device,
                positive_prompts=[
                    "butterfly wing silhouette",
                    "black and white",
                    "high contrast",
                    "minimal",
                    "outline",
                ],
                negative_prompts=[
                    "photograph",
                    "texture",
                    "shading",
                    "noise",
                    "background",
                    "multiple objects",
                    "text",
                    "watermark",
                ],
                num_augs=24,
                preblur_sigma=0.7,
                num_global_views=1,
                center_crop_frac=0.90,
            )

        params = StructuralParams(
            problem_name="multistory_building",
            width=50,
            height=100,
            density=0.3,
            num_stories=5,
            # Filter radius in element units, scaled with the grid on each
            # upsample. A radius <= 1.0 degenerates the cone filter to the
            # identity, so this is the floor that still filters.
            rmin=1.0,
            # Resolves to a fixed 2 * rmin, the legacy 2.0px radius. There is no
            # within-stage schedule; continuation is not implemented yet.
            filter_width="linear",
            # Resolves to a fixed beta=1.0; no sharpening schedule.
            beta="linear" 
        )
        params, dynamic_kwargs = pipeline_utils.dynamic_depth_kwargs(params)

        print("Dynamic kwargs:")
        for key, value in dynamic_kwargs.items():
            print(f"  {key}: {value}")

        print(f"Problem: {params.problem_name}")
        print(f"Dimensions: {params.width}x{params.height}")
        print(f"Max iterations: {max_iterations}")

        model = models.CNNModel(structural_params=params, clip_loss=clip_loss, **dynamic_kwargs)
        # Do not compile the CNN during progressive resizing; it interferes with upsampling
        trainer = PixelRefineTrainer(
            model,
            max_iterations,
            resize_num=4,
            switch_threshold=500,
            coarse_start=True,
            initial_image=None,
        )
        # Match pasted-script-style parameters:
        #   - clip_alpha: scales semantic CLIP by current compliance (detached)
        #   - compliance_weight: scales the structural/compliance loss
        ds_history = normalize_history(trainer.train(
            LBFGS_Optimizer,
            clip_alpha=1e-2,
            compliance_weight=1.0,
            # You can still use warmup/static weighting if clip_alpha=None
            clip_weight_max=1.0,
            clip_warmup_steps=0,
        ))

        print(f"\nOptimization completed! Stages: {len(ds_history)}")

        print("\nCreating and saving plots...")
        loss_plot_path = save_loss_plot(ds_history, filename_suffix)
        designs_plot_path = save_final_designs_plot(ds_history, params, filename_suffix, force_legacy=force_legacy_rendering)

        print("All plots saved successfully!")
        print(f"Loss plot: {loss_plot_path}")
        print(f"Designs plot: {designs_plot_path}")

    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0



if __name__ == "__main__":
    exit(main())