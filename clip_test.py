# Example 1: Text-guided optimization
from neural_structural_optimization import problems, models, clip_config

# Create CLIP configuration for text prompt
clip_cfg = clip_config.create_clip_config(
    text_prompt="a bridge with elegant curved arches",
    weight=0.2
)

# Create problem and model
problem = problems.cantilever_beam_full(width=60, height=60, density=0.5)
model = models.CNNModel(args=problem, clip_config=clip_cfg)

# Train as usual
from neural_structural_optimization import train
results = train.train_adam(model, max_iterations=1000)

# Example 2: Image-guided optimization
clip_cfg = clip_config.create_clip_config(
    target_image_path="path/to/reference_bridge.jpg",
    weight=0.15
)

model = models.CNNModel(args=problem, clip_config=clip_cfg)
results = train.train_adam(model, max_iterations=1000)