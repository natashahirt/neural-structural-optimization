"""Problem definitions and utilities for structural optimization.

This module consolidates problem definitions, parameter handling, and problem utilities
for structural optimization tasks.
"""

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

"""
overview:
- multiple different topopt problems
- Problem dataclass includes boundary conditions, forces, constraints
- StructuralParams dataclass for parameterized problem creation
"""

import inspect
from typing import Optional, Union

import dataclasses

import numpy as np
import skimage.draw


X, Y = 0, 1


@dataclasses.dataclass
class Problem:
  """Description of a topology optimization problem.

  Attributes:
    normals: float64 array of shape (width+1, height+1, 2) where a value of 1
      indicates a "fixed" coordinate, and 0 indicates no normal force.
    forces: float64 array of shape (width+1, height+1, 2) indicating external
      applied forces in the x and y directions.
    density: fraction of the design region that should be non-zero.
    mask: scalar or float64 array of shape (height, width) that is multiplied by
      the design mask before and after applying the blurring filters. Values of
      1 indicate regions where the material can be optimized; values of 0 are
      constrained to be empty.
    name: optional name of this problem.
    width: integer width of the domain.
    height: integer height of the domain.
    mirror_left: should the design be mirrored to the left when displayed?
    mirror_right: should the design be mirrored to the right when displayed?
  """
  normals: np.ndarray
  forces: np.ndarray
  density: float
  mask: Union[np.ndarray, float] = 1
  name: Optional[str] = None
  width: int = dataclasses.field(init=False)
  height: int = dataclasses.field(init=False)
  mirror_left: bool = dataclasses.field(init=False)
  mirror_right: bool = dataclasses.field(init=False)

  def __post_init__(self):
    self.width = self.normals.shape[0] - 1
    self.height = self.normals.shape[1] - 1

    if self.normals.shape != (self.width + 1, self.height + 1, 2):
      raise ValueError(f'normals has wrong shape: {self.normals.shape}')
    if self.forces.shape != (self.width + 1, self.height + 1, 2):
      raise ValueError(f'forces has wrong shape: {self.forces.shape}')
    if (isinstance(self.mask, np.ndarray)
        and self.mask.shape != (self.height, self.width)):
      raise ValueError(f'mask has wrong shape: {self.mask.shape}')

    self.mirror_left = (
        self.normals[0, :, X].all() and not self.normals[0, :, Y].all()
    )
    self.mirror_right = (
        self.normals[-1, :, X].all() and not self.normals[-1, :, Y].all()
    )

  def copy(self, width=None, height=None, density=None, name=None):
    new_width = width if width is not None else self.width
    new_height = height if height is not None else self.height
    new_density = density if density is not None else self.density
    new_name = name if name is not None else self.name

    # Initialize new arrays
    new_normals = np.zeros((new_width + 1, new_height + 1, 2))
    new_forces = np.zeros((new_width + 1, new_height + 1, 2))

    # Area ratio for force scaling
    area_ratio = (self.width * self.height) / (new_width * new_height)

    for i in range(new_width + 1):
        for j in range(new_height + 1):
            # Map new node (i, j) to original grid
            orig_i = int(round(i * self.width / new_width))
            orig_j = int(round(j * self.height / new_height))

            # Clamp to bounds
            orig_i = min(max(orig_i, 0), self.width)
            orig_j = min(max(orig_j, 0), self.height)

            new_normals[i, j, :] = self.normals[orig_i, orig_j, :]
            new_forces[i, j, :] = self.forces[orig_i, orig_j, :] * area_ratio

    # Resize mask
    if isinstance(self.mask, np.ndarray):
        new_mask = np.ones((new_height, new_width), dtype=self.mask.dtype)
        for i in range(new_width):
            for j in range(new_height):
                orig_i = int(round(i * self.width / new_width))
                orig_j = int(round(j * self.height / new_height))
                orig_i = min(max(orig_i, 0), self.width - 1)
                orig_j = min(max(orig_j, 0), self.height - 1)
                new_mask[j, i] = self.mask[orig_j, orig_i]
    else:
        new_mask = self.mask

    return Problem(new_normals, new_forces, new_density, new_mask, new_name)    


def mbb_beam(width=60, height=20, density=0.5):
  """Textbook beam example."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[-1, -1, Y] = 1
  normals[0, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[0, 0, Y] = -1

  problem = Problem(normals, forces, density)
  problem.name = f"mbb_beam_{width}x{height}"
  return problem


def cantilever_beam_full(
    width=60, height=60, density=0.5, force_position=0):
  """Cantilever supported everywhere on the left."""
  # https://link.springer.com/content/pdf/10.1007%2Fs00158-010-0557-z.pdf
  normals = np.zeros((width + 1, height + 1, 2))
  normals[0, :, :] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[-1, round((1 - force_position)*height), Y] = -1

  problem = Problem(normals, forces, density)
  problem.name = f"cantilever_beam_full_{width}x{height}"
  return problem


def cantilever_beam_two_point(
    width=60, height=60, density=0.5, support_position=0.25,
    force_position=0.5):
  """Cantilever supported by two points."""
  # https://link.springer.com/content/pdf/10.1007%2Fs00158-010-0557-z.pdf
  normals = np.zeros((width + 1, height + 1, 2))
  normals[0, round(height*(1-support_position)), :] = 1
  normals[0, round(height*support_position), :] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[-1, round((1 - force_position)*height), Y] = -1

  problem = Problem(normals, forces, density)
  problem.name = f"cantilever_beam_two_point_{width}x{height}"
  return problem


def pure_bending_moment(
    width=60, height=60, density=0.5, support_position=0.45):
  """Pure bending forces on a beam."""
  # Figure 28 from
  # http://naca.central.cranfield.ac.uk/reports/arc/rm/3303.pdf
  normals = np.zeros((width + 1, height + 1, 2))
  normals[-1, :, X] = 1
  # for numerical stability, fix y forces here at 0
  normals[0, round(height*(1-support_position)), Y] = 1
  normals[0, round(height*support_position), Y] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[0, round(height*(1-support_position)), X] = 1
  forces[0, round(height*support_position), X] = -1

  problem = Problem(normals, forces, density)
  problem.name = f"pure_bending_moment_{width}x{height}"
  return problem


def michell_centered_both(width=32, height=32, density=0.5, position=0.05):
  """A single force down at the center, with support from the side."""
  # https://en.wikipedia.org/wiki/Michell_structures#Examples
  normals = np.zeros((width + 1, height + 1, 2))
  normals[round(position*width), round(height/2), Y] = 1
  normals[-1, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[-1, round(height/2), Y] = -1

  problem = Problem(normals, forces, density)
  problem.name = f"michell_centered_both_{width}x{height}"
  return problem


def michell_centered_below(width=32, height=32, density=0.5, position=0.25):
  """A single force down at the center, with support from the side below."""
  # https://en.wikipedia.org/wiki/Michell_structures#Examples
  normals = np.zeros((width + 1, height + 1, 2))
  normals[round(position*width), 0, Y] = 1
  normals[-1, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[-1, 0, Y] = -1

  problem = Problem(normals, forces, density)
  problem.name = f"michell_centered_below_{width}x{height}"
  return problem


def ground_structure(width=32, height=32, density=0.5, force_position=0.5):
  """An overhanging bridge like structure holding up two weights."""
  # https://link.springer.com/content/pdf/10.1007%2Fs00158-010-0557-z.pdf
  normals = np.zeros((width + 1, height + 1, 2))
  normals[-1, :, X] = 1
  normals[0, -1, :] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[round(force_position*height), -1, Y] = -1

  problem = Problem(normals, forces, density)
  problem.name = f"ground_structure_{width}x{height}"
  return problem


def l_shape(width=32, height=32, density=0.5, aspect=0.4, force_position=0.5):
  """An L-shaped structure, with a limited design region."""
  # Topology Optimization Benchmarks in 2D
  normals = np.zeros((width + 1, height + 1, 2))
  normals[:round(aspect*width), 0, :] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[-1, round((1 - aspect*force_position)*height), Y] = -1

  mask = np.ones((width, height))
  mask[round(height*aspect):, :round(width*(1-aspect))] = 0

  problem = Problem(normals, forces, density, mask.T)
  problem.name = f"l_shape_{width}x{height}"
  return problem


def crane(width=32, height=32, density=0.3, aspect=0.5, force_position=0.9):
  """A crane supporting a downward force, anchored on the left."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[:, -1, :] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[round(force_position*width), round(1-aspect*height), Y] = -1

  mask = np.ones((width, height))
  # the extra +2 ensures that entire region in the vicinity of the force can be
  # be designed; otherwise we get outrageously high values for the compliance.
  mask[round(aspect*width):, round(height*aspect)+2:] = 0

  problem = Problem(normals, forces, density, mask.T)
  problem.name = f"crane_{width}x{height}"
  return problem


def tower(width=32, height=32, density=0.5):
  """A rather boring structure supporting a single point from the ground."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[:, -1, Y] = 1
  normals[0, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[0, 0, Y] = -1

  problem = Problem(normals, forces, density)
  problem.name = f"tower_{width}x{height}"
  return problem


def center_support(width=32, height=32, density=0.3):
  """Support downward forces from the top from the single point."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[-1, -1, Y] = 1
  normals[-1, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[:, 0, Y] = -1 / width

  problem = Problem(normals, forces, density)
  problem.name = f"center_support_{width}x{height}"
  return problem


def column(width=32, height=32, density=0.3):
  """Support downward forces from the top across a finite width."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[:, -1, Y] = 1
  normals[-1, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[:, 0, Y] = -1 / width

  problem = Problem(normals, forces, density)
  problem.name = f"column_{width}x{height}"
  return problem


def roof(width=32, height=32, density=0.5):
  """Support downward forces from the top with a repeating structure."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[0, :, X] = 1
  normals[-1, :, X] = 1
  normals[:, -1, Y] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[:, 0, Y] = -1 / width

  problem = Problem(normals, forces, density)
  problem.name = f"roof_{width}x{height}"
  return problem


def causeway_bridge(width=60, height=20, density=0.3, deck_level=1):
  """A bridge supported by columns at a regular interval."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[-1, -1, Y] = 1
  normals[-1, :, X] = 1
  normals[0, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[:, round(height * (1 - deck_level)), Y] = -1 / width

  problem = Problem(normals, forces, density)
  problem.name = f"causeway_bridge_{width}x{height}"
  return problem


def two_level_bridge(width=32, height=32, density=0.3, deck_height=0.2):
  """A causeway bridge with two decks."""
  normals = np.zeros((width + 1, width + 1, 2))
  normals[0, -1, :] = 1
  normals[0, :, X] = 1
  normals[-1, :, X] = 1

  forces = np.zeros((width + 1, width + 1, 2))
  forces[:, round(height * (1 - deck_height) / 2), :] = -1 / (2 * width)
  forces[:, round(height * (1 + deck_height) / 2), :] = -1 / (2 * width)

  problem = Problem(normals, forces, density)
  problem.name = f"two_level_bridge_{width}x{height}"
  return problem


def suspended_bridge(width=60, height=20, density=0.3, span_position=0.2,
                     anchored=False):
  """A bridge above the ground, with supports at lower corners."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[-1, :, X] = 1
  normals[:round(span_position*width), -1, Y] = 1
  if anchored:
    normals[:round(span_position*width), -1, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[:, -1, Y] = -1 / width

  problem = Problem(normals, forces, density)
  problem.name = f"suspended_bridge_{width}x{height}"
  return problem


def canyon_bridge(width=60, height=20, density=0.3, deck_level=1):
  """A bridge embedded in a canyon, without side supports."""
  deck_height = round(height * (1 - deck_level))

  normals = np.zeros((width + 1, height + 1, 2))
  normals[-1, deck_height:, :] = 1
  normals[0, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[:, deck_height, Y] = -1 / width

  problem = Problem(normals, forces, density)
  problem.name = f"canyon_bridge_{width}x{height}"
  return problem


def thin_support_bridge(
    width=32, height=32, density=0.25, design_width=0.25):
  """A bridge supported from below with fixed width supports."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[:, -1, Y] = 1
  normals[0, :, X] = 1
  normals[-1, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[:, 0, Y] = -1 / width

  mask = np.ones((width, height))
  mask[-round(width*(1-design_width)):, :round(height*(1-design_width))] = 0

  problem = Problem(normals, forces, density, mask)
  problem.name = f"thin_support_bridge_{width}x{height}"
  return problem


def drawbridge(width=32, height=32, density=0.25):
  """A bridge supported from above on the left."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[0, :, :] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[:, -1, Y] = -1 / width

  problem = Problem(normals, forces, density)
  problem.name = f"drawbridge_{width}x{height}"
  return problem


def hoop(width=32, height=32, density=0.25):
  """Downward forces in a circle, supported from the ground."""
  if 2 * width != height:
    raise ValueError('hoop must be circular')

  normals = np.zeros((width + 1, height + 1, 2))
  normals[-1, :, X] = 1
  normals[:, -1, Y] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  i, j, value = skimage.draw.circle_perimeter_aa(
      width, width, width, forces.shape[:2]
  )
  forces[i, j, Y] = -value / (2 * np.pi * width)

  problem = Problem(normals, forces, density)
  problem.name = f"hoop_{width}x{height}"
  return problem


def multipoint_circle(
    width=140, height=140, density=0.333, radius=6/7,
    weights=(1, 0, 0, 0, 0, 0), num_points=12):
  """Various load scenarios at regular points in a circle points."""
  # From: http://www2.mae.ufl.edu/mdo/Papers/5219.pdf
  # Note: currently unused in our test suite only because the optimization
  # problems from the paper are defined based on optimizing for compliance
  # averaged over multiple force scenarios.
  c_x = width // 2
  c_y = height // 2
  normals = np.zeros((width + 1, height + 1, 2))
  normals[c_x - 1 : c_x + 2, c_y - 1 : c_y + 2, :] = 1
  assert normals.sum() == 18

  c1, c2, c3, c4, c_x0, c_y0 = weights

  forces = np.zeros((width + 1, height + 1, 2))
  for position in range(num_points):
    x = radius * c_x * np.sin(2*np.pi*position/num_points)
    y = radius * c_y * np.cos(2*np.pi*position/num_points)
    i = int(round(c_x + x))
    j = int(round(c_y + y))
    forces[i, j, X] = + c1 * y + c2 * x + c3 * y + c4 * x + c_x0
    forces[i, j, Y] = - c1 * x + c2 * y + c3 * x - c4 * y + c_y0

  problem = Problem(normals, forces, density)
  problem.name = f"multipoint_circle_{width}x{height}"
  return problem


def dam(width=32, height=32, density=0.5):
  """Support horizitonal forces, proportional to depth."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[:, -1, X] = 1
  normals[:, -1, Y] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  forces[0, :, X] = 2 * np.arange(1, height+2) / height ** 2

  problem = Problem(normals, forces, density)
  problem.name = f"dam_{width}x{height}"
  return problem


def ramp(width=32, height=32, density=0.25):
  """Support downward forces on a ramp."""
  problem = staircase(width, height, density, num_stories=1)
  problem.name = f"ramp_{width}x{height}"
  return problem


def staircase(width=32, height=32, density=0.25, num_stories=2):
  """A ramp that zig-zags upward, supported from the ground."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[:, -1, :] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  for story in range(num_stories):
    parity = story % 2
    start_coordinates = (0, (story + parity) * height // num_stories)
    stop_coordiates = (width, (story + 1 - parity) * height // num_stories)
    i, j, value = skimage.draw.line_aa(*start_coordinates, *stop_coordiates)
    forces[i, j, Y] = np.minimum(
        forces[i, j, Y], -value / (width * num_stories)
    )

  problem = Problem(normals, forces, density)
  problem.name = f"staircase_{width}x{height}"
  return problem


def staggered_points(width=32, height=32, density=0.3, interval=16,
                     break_symmetry=False):
  """A staggered grid of points with downward forces, supported from below."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[:, -1, Y] = 1
  normals[0, :, X] = 1
  normals[-1, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))
  f = interval ** 2 / (width * height)
  # intentionally break horizontal symmetry?
  forces[interval//2+int(break_symmetry)::interval, ::interval, Y] = -f
  forces[int(break_symmetry)::interval, interval//2::interval, Y] = -f

  problem = Problem(normals, forces, density)
  problem.name = f"staggered_points_{width}x{height}"
  return problem


def multistory_building(width=32, height=32, density=0.3, num_stories=16):
  """A multi-story building, supported from the ground."""
  normals = np.zeros((width + 1, height + 1, 2))
  normals[:, -1, Y] = 1
  normals[-1, :, X] = 1

  forces = np.zeros((width + 1, height + 1, 2))

  story_height = height // num_stories
  for story in range(0, num_stories):
    story_y = story * story_height
    forces[:, story_y, Y] = -1 / width

  problem = Problem(normals, forces, density)
  problem.name = f"multistory_building_{width}x{height}"
  return problem

# =============================================================================
# StructuralParams dataclass for parameterized problem creation
# =============================================================================

@dataclasses.dataclass
class StructuralParams:
    """
    Parameterized problem creation utility.
    
    Example use: 
    
    # Create problem parameters
    params = StructuralParams(
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
    
    # filtering parameters
    filter_width: float = 1.5  # filter width for density filtering
    rmin: float = 1.5  # minimum radius for density filtering
    
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
        
        # Validate filtering parameters
        if self.filter_width <= 0.0:
            raise ValueError(f"filter_width must be positive, got {self.filter_width}")
        if self.rmin <= 0.0:
            raise ValueError(f"rmin must be positive, got {self.rmin}")
        
        # Validate special cases
        if self.problem_name == "hoop" and 2 * self.width != self.height:
            raise ValueError("hoop problems require height = 2 * width")
    
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def copy(self, **kwargs) -> 'StructuralParams':
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

# pylint: disable=line-too-long
PROBLEMS_BY_CATEGORY = {
    # idealized beam and cantilevers
    'mbb_beam': [
        mbb_beam(96, 32, density=0.5),
        mbb_beam(192, 64, density=0.4),
        mbb_beam(384, 128, density=0.3),
        mbb_beam(192, 32, density=0.5),
        mbb_beam(384, 64, density=0.4),
    ],
    'cantilever_beam_full': [
        cantilever_beam_full(96, 32, density=0.4),
        cantilever_beam_full(192, 64, density=0.3),
        cantilever_beam_full(384, 128, density=0.2),
        cantilever_beam_full(384, 128, density=0.15),
    ],
    'cantilever_beam_two_point': [
        cantilever_beam_two_point(64, 48, density=0.4),
        cantilever_beam_two_point(128, 96, density=0.3),
        cantilever_beam_two_point(256, 192, density=0.2),
        cantilever_beam_two_point(256, 192, density=0.15),
    ],
    'pure_bending_moment': [
        pure_bending_moment(32, 64, density=0.15),
        pure_bending_moment(64, 128, density=0.125),
        pure_bending_moment(128, 256, density=0.1),
    ],
    'ground_structure': [
        ground_structure(64, 64, density=0.12),
        ground_structure(128, 128, density=0.1),
        ground_structure(256, 256, density=0.07),
        ground_structure(256, 256, density=0.05),
    ],
    'michell_centered_both': [
        michell_centered_both(32, 64, density=0.12),
        michell_centered_both(64, 128, density=0.12),
        michell_centered_both(128, 256, density=0.12),
        michell_centered_both(128, 256, density=0.06),
    ],
    'michell_centered_below': [
        michell_centered_below(64, 64, density=0.12),
        michell_centered_below(128, 128, density=0.12),
        michell_centered_below(256, 256, density=0.12),
        michell_centered_below(256, 256, density=0.06),
    ],
    # simple constrained designs
    'l_shape_0.2': [
        l_shape(64, 64, aspect=0.2, density=0.4),
        l_shape(128, 128, aspect=0.2, density=0.3),
        l_shape(256, 256, aspect=0.2, density=0.2),
    ],
    'l_shape_0.4': [
        l_shape(64, 64, aspect=0.4, density=0.4),
        l_shape(128, 128, aspect=0.4, density=0.3),
        l_shape(256, 256, aspect=0.4, density=0.2),
    ],
    'crane': [
        crane(64, 64, density=0.3),
        crane(128, 128, density=0.2),
        crane(256, 256, density=0.15),
        crane(256, 256, density=0.1),
    ],
    # vertical support structures
    'center_support': [
        center_support(64, 64, density=0.15),
        center_support(128, 128, density=0.1),
        center_support(256, 256, density=0.1),
        center_support(256, 256, density=0.05),
    ],
    'column': [
        column(32, 128, density=0.3),
        column(64, 256, density=0.3),
        column(128, 512, density=0.1),
        column(128, 512, density=0.3),
        column(128, 512, density=0.5),
    ],
    'roof': [
        roof(64, 64, density=0.2),
        roof(128, 128, density=0.15),
        roof(256, 256, density=0.4),
        roof(256, 256, density=0.2),
        roof(256, 256, density=0.1),
    ],
    # bridges
    'causeway_bridge_top': [
        causeway_bridge(64, 64, density=0.3),
        causeway_bridge(128, 128, density=0.2),
        causeway_bridge(256, 256, density=0.1),
        causeway_bridge(128, 64, density=0.3),
        causeway_bridge(256, 128, density=0.2),
    ],
    'causeway_bridge_middle': [
        causeway_bridge(64, 64, density=0.12, deck_level=0.5),
        causeway_bridge(128, 128, density=0.1, deck_level=0.5),
        causeway_bridge(256, 256, density=0.08, deck_level=0.5),
    ],
    'causeway_bridge_low': [
        causeway_bridge(64, 64, density=0.12, deck_level=0.3),
        causeway_bridge(128, 128, density=0.1, deck_level=0.3),
        causeway_bridge(256, 256, density=0.08, deck_level=0.3),
    ],
    'two_level_bridge': [
        two_level_bridge(64, 64, density=0.2),
        two_level_bridge(128, 128, density=0.16),
        two_level_bridge(256, 256, density=0.12),
    ],
    'free_suspended_bridge': [
        suspended_bridge(64, 64, density=0.15, anchored=False),
        suspended_bridge(128, 128, density=0.1, anchored=False),
        suspended_bridge(256, 256, density=0.075, anchored=False),
        suspended_bridge(256, 256, density=0.05, anchored=False),
    ],
    'anchored_suspended_bridge': [
        suspended_bridge(64, 64, density=0.15, span_position=0.1, anchored=True),
        suspended_bridge(128, 128, density=0.1, span_position=0.1, anchored=True),
        suspended_bridge(256, 256, density=0.075, span_position=0.1, anchored=True),
        suspended_bridge(256, 256, density=0.05, span_position=0.1, anchored=True),
    ],
    'canyon_bridge': [
        canyon_bridge(64, 64, density=0.16),
        canyon_bridge(128, 128, density=0.12),
        canyon_bridge(256, 256, density=0.1),
        canyon_bridge(256, 256, density=0.05),
    ],
    'thin_support_bridge': [
        thin_support_bridge(64, 64, density=0.3),
        thin_support_bridge(128, 128, density=0.2),
        thin_support_bridge(256, 256, density=0.15),
        thin_support_bridge(256, 256, density=0.1),
    ],
    'drawbridge': [
        drawbridge(64, 64, density=0.2),
        drawbridge(128, 128, density=0.15),
        drawbridge(256, 256, density=0.1),
    ],
    # more complex design problems
    'hoop': [
        hoop(32, 64, density=0.25),
        hoop(64, 128, density=0.2),
        hoop(128, 256, density=0.15),
    ],
    'dam': [
        dam(64, 64, density=0.2),
        dam(128, 128, density=0.15),
        dam(256, 256, density=0.05),
        dam(256, 256, density=0.1),
        dam(256, 256, density=0.2),
    ],
    'ramp': [
        ramp(64, 64, density=0.3),
        ramp(128, 128, density=0.2),
        ramp(256, 256, density=0.2),
        ramp(256, 256, density=0.1),
    ],
    'staircase': [
        staircase(64, 64, density=0.3, num_stories=3),
        staircase(128, 128, density=0.2, num_stories=3),
        staircase(256, 256, density=0.15, num_stories=3),
        staircase(128, 512, density=0.15, num_stories=6),
    ],
    'staggered_points': [
        staggered_points(64, 64, density=0.3),
        staggered_points(128, 128, density=0.3),
        staggered_points(256, 256, density=0.3),
        staggered_points(256, 256, density=0.5),
        staggered_points(64, 128, density=0.3),
        staggered_points(128, 256, density=0.3),
        staggered_points(32, 128, density=0.3),
        staggered_points(64, 256, density=0.3),
        staggered_points(128, 512, density=0.3),
        staggered_points(128, 512, interval=32, density=0.15),
    ],
    'multistory_building': [
        multistory_building(32, 64, density=0.5),
        multistory_building(64, 128, num_stories=32, density=0.4),
        multistory_building(128, 256, num_stories=64, density=0.3),
        multistory_building(128, 512, num_stories=64, density=0.25),
        multistory_building(128, 512, num_stories=128, density=0.2),
    ],
}

PROBLEMS_BY_NAME = {}
for problem_class, problem_list in PROBLEMS_BY_CATEGORY.items():
  for problem in problem_list:
    name = f'{problem_class}_{problem.width}x{problem.height}_{problem.density}'
    problem.name = name
    assert name not in PROBLEMS_BY_NAME, f'redundant name {name}'
    PROBLEMS_BY_NAME[name] = problem