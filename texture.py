"""
Texture Sampling Template - Student Implementation Required

Implements texture sampling and filtering for image elements.
Students must implement the functions marked with TODO comments.

Tasks in this file:
- Task 4: Image Rasterization
  - sample_nearest() - Nearest neighbor texture sampling
  - sample_bilinear() - Bilinear texture filtering
  - sample_trilinear() - Trilinear filtering with mipmap interpolation
  - generate_mips() - Mipmap generation using box filtering
"""

import numpy as np
from math_utils import Color, clamp, lerp


class Sampler2D:
    """2D texture sampler with multiple filtering modes."""
    
    def __init__(self, texture_data: np.ndarray):
        """Initialize sampler with texture data."""
        self.texture = texture_data
        self.width = texture_data.shape[1]
        self.height = texture_data.shape[0]
        self.mipmaps = [texture_data]
    
    def set_mipmaps(self, mipmaps: list):
        """Set mipmap levels."""
        self.mipmaps = mipmaps
    
    def sample_nearest(self, u: float, v: float, level: int = 0) -> Color:
        """Sample texture using nearest neighbor filtering."""
        """
        TASK 4: Image Rasterization - Nearest Neighbor Sampling
        
        TODO: Implement nearest neighbor texture sampling.
        
        Requirements:
        - Clamp UV coordinates to [0,1] range
        - Select appropriate mipmap level
        - Convert UV to pixel coordinates
        - Round to nearest pixel (nearest neighbor)
        - Return sampled color
        
        Steps:
        1. Clamp u,v to [0,1] range using clamp() function
        2. Select mipmap level (clamp level to available mipmaps)
        3. Convert UV to pixel coordinates: x = u * (width-1), y = v * (height-1)
        4. Round to nearest integer pixel coordinates
        5. Clamp pixel coordinates to texture bounds
        6. Sample pixel and return as Color
        
        Note: UV (0,0) = top-left pixel, UV (1,1) = bottom-right pixel
        """

        pass  # Remove this line when implementing
    
    def sample_bilinear(self, u: float, v: float, level: int = 0) -> Color:
        """Sample texture using bilinear filtering."""
        """
        TASK 4: Image Rasterization - Bilinear Filtering
        
        TODO: Implement bilinear texture sampling with proper edge handling.
        
        Requirements:
        - Sample four surrounding pixels and interpolate between them
        - Handle edge clamping (GL_CLAMP_TO_EDGE behavior)
        - Use fractional pixel coordinates for smooth interpolation
        - Return interpolated color
        
        Steps:
        1. Clamp UV coordinates to [0,1] range
        2. Convert UV to pixel coordinates (offset by 0.5 for pixel centers)
        3. Get integer and fractional parts of pixel coordinates
        4. Find four surrounding pixels with clamping to texture bounds
        5. Sample all four pixels as float values
        6. Perform bilinear interpolation:
           - Interpolate horizontally: top = p00*(1-fx) + p10*fx, bottom = p01*(1-fx) + p11*fx
           - Interpolate vertically: final = top*(1-fy) + bottom*fy
        7. Return interpolated color
        
        Bilinear interpolation smooths texture sampling by blending nearby pixels.
        """
        
        pass  # Remove this line when implementing
    
    def sample_trilinear(self, u: float, v: float, mip_level: float) -> Color:
        """Sample texture using trilinear filtering."""
        """
        TASK 4: Trilinear Filtering
        
        TODO: Implement trilinear filtering using mipmap interpolation.
        
        Requirements:
        - Sample from two adjacent mipmap levels
        - Interpolate between the two samples based on fractional mip level
        - Use bilinear sampling for each mipmap level
        - Handle edge cases (single mip level, fractional near 0)
        
        Steps:
        1. Clamp mip_level to valid range [0, num_mipmaps-1]
        2. Get integer and fractional parts of mip level
        3. If fractional part is near 0 or at highest level, use bilinear only
        4. Sample from two adjacent mip levels using bilinear sampling
        5. Interpolate between the two colors using linear interpolation (lerp)
        
        Trilinear filtering reduces aliasing when textures appear at different scales.
        """

        pass  # Remove this line when implementing
    


class Sampler2DImp(Sampler2D):
    """Implementation class for Sampler2D with mipmap generation."""
    
    def generate_mips(self):
        """Generate mipmap hierarchy for the texture."""
        """
        TASK 4: Mipmap Generation (Part of Trilinear Filtering)
        
        TODO: Generate mipmap hierarchy using box filtering.
        
        Requirements:
        - Generate progressively smaller versions of texture (each level half the size)
        - Use box filtering (average 2x2 blocks) to downsample
        - Handle edge cases for non-power-of-two textures
        - Store mipmaps in self.mipmaps list
        
        Steps:
        1. Check if texture has valid size
        2. Start with original texture as level 0
        3. For each level until 1x1 texture:
           - Calculate new dimensions (half width/height, minimum 1)
           - Create new mipmap array
           - For each pixel in new level, average corresponding 2x2 block from previous level
           - Handle edge cases where source coordinates exceed bounds
        4. Store all mipmap levels in self.mipmaps
        
        Box filtering reduces aliasing by pre-averaging texture details at multiple scales.
        """
        
        pass  # Remove this line when implementing


def uint8_to_float(value: int) -> float:
    """Convert uint8 color value to float [0, 1] range."""
    return value / 255.0


def float_to_uint8(value: float) -> int:
    """Convert float color value to uint8 [0, 255] range."""
    return int(clamp(value * 255.0, 0, 255))