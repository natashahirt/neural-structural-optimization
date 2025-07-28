import inspect
from neural_structural_optimization.problems import *

# =============================================================================
# Params dataclass
# =============================================================================

@dataclasses.dataclass
class ProblemParams:
    """
    example use: 
    
    # Create problem parameters
    params = ProblemParams(
        problem_name="cantilever_beam_full",
        width=60,
        height=60,
        density=0.4,
        force_position=0.5
    )

    # Get the problem and create a model
    problem = params.get_problem()
    """
    
    # general
    problem_name: str = "cantilever_beam_full"
    width: int = 60
    height: int = 60
    density: float = 0.5

    # for beam and cantilever
    force_position: float = 0.5 # 0. is top, 1. is bottom
    support_position: float = 0.25 # for 2-point cantilevers

    # for bridge
    deck_level: float = 1. # for causeway bridges, 0. is top, 1. is bottom
    deck_height: float = 0.2 # for two-level bridges, 0. is top, 1. is bottom
    span_position: float = 0.2 # for suspended bridges
    anchored: bool = False # is suspended bridge anchored?
    design_width: float = 0.25 # for thin support bridges

    # for shape/aspect
    aspect: float = 0.4 # for L-shaped/crane problems

    # for multipoint circle only
    radius: float = 6/7
    weights: tuple = (1,)  # Single weight for one point
    num_points: int = 1

    # n_stories for staircase
    num_stories: int = 2

    # grid and interval params
    interval: int = 16 # for multistory buildings
    break_symmetry: bool = False # for staggered points

    # position for michell_centered_both
    position: float = 0.05

    # validation + utility
    def __post_init__(self):
        if not 0.0 < self.density <= 1.0:
            raise ValueError(f"density must be positive, nonzero, between 0. and 1. Got {self.density}.")
        
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width and height must be greater than 0. Got {self.width}x{self.height}.")

        for param_name in ['force_position', 'support_position', 'deck_level', 
                          'deck_height', 'span_position', 'design_width', 
                          'aspect', 'radius', 'position']:
            value = getattr(self, param_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{param_name} must be between 0 and 1, got {value}")
        
        # Validate special cases
        if self.problem_name == "hoop" and 2 * self.width != self.height:
            raise ValueError("hoop problems require height = 2 * width")
    
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def copy(self, **kwargs) -> 'ProblemParams':
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def get_available_problems(cls) -> list:
        """Get list of all available problem names."""
        return [
            # Beam and cantilever problems
            "mbb_beam",
            "cantilever_beam_full", 
            "cantilever_beam_two_point",
            "pure_bending_moment",
            
            # Michell structures
            "michell_centered_both",
            "michell_centered_below",
            
            # Constrained designs
            "ground_structure",
            "l_shape",
            "crane",
            
            # Vertical support structures
            "tower",
            "center_support", 
            "column",
            "roof",
            
            # Bridge problems
            "causeway_bridge",
            "two_level_bridge",
            "suspended_bridge",
            "canyon_bridge",
            "thin_support_bridge",
            "drawbridge",
            
            # Complex designs
            "hoop",
            "multipoint_circle",
            "dam",
            "ramp",
            "staircase",
            "staggered_points",
            "multistory_building"
        ]

    def get_problem(self) -> 'Problem':
        problem_function = globals().get(self.problem_name)
        if problem_function is None or not callable(problem_function):
            raise ValueError(f"No problem found with the name {self.problem_name}.")

        sig = inspect.signature(problem_function)
        filtered_params = {k: v for k, v in self.to_dict().items() if k in sig.parameters}
        return problem_function(**filtered_params)