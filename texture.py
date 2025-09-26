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
        # Clamp UV
        u = clamp(u, 0.0, 1.0)
        v = clamp(v, 0.0, 1.0)

        # Clamp mipmap level
        level = int(clamp(level, 0, len(self.mipmaps) - 1))
        tex = self.mipmaps[level]
        h, w = tex.shape[0], tex.shape[1]

        # Convert UV to pixel coordinates
        x = int(round(u * (w - 1)))
        y = int(round(v * (h - 1)))

        # Clamp pixel coords
        x = clamp(x, 0, w - 1)
        y = clamp(y, 0, h - 1)

        # Sample pixel (RGBA in [0,255])
        r, g, b, a = tex[y, x]

        # Convert to Color (float 0-1)
        return Color(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    
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
        # Clamp UV
        u = clamp(u, 0.0, 1.0)
        v = clamp(v, 0.0, 1.0)

        # Clamp mip level
        level = int(clamp(level, 0, len(self.mipmaps) - 1))
        tex = self.mipmaps[level]
        h, w = tex.shape[0], tex.shape[1]

        # Pixel coordinates (centered)
        x = u * (w - 1)
        y = v * (h - 1)

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        fx = x - x0
        fy = y - y0

        # Sample four pixels
        c00 = tex[y0, x0]  # top-left
        c10 = tex[y0, x1]  # top-right
        c01 = tex[y1, x0]  # bottom-left
        c11 = tex[y1, x1]  # bottom-right

        # Convert to float
        c00 = [c / 255.0 for c in c00]
        c10 = [c / 255.0 for c in c10]
        c01 = [c / 255.0 for c in c01]
        c11 = [c / 255.0 for c in c11]

        # Interpolate
        top = [lerp(c00[i], c10[i], fx) for i in range(4)]
        bottom = [lerp(c01[i], c11[i], fx) for i in range(4)]
        final = [lerp(top[i], bottom[i], fy) for i in range(4)]

        return Color(*final)
    
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
        num_levels = len(self.mipmaps)
        mip_level = clamp(mip_level, 0.0, num_levels - 1.0)

        l0 = int(np.floor(mip_level))
        l1 = min(l0 + 1, num_levels - 1)
        t = mip_level - l0  # fractional part

        c0 = self.sample_bilinear(u, v, l0)
        if t < 1e-6 or l0 == l1:
            return c0

        c1 = self.sample_bilinear(u, v, l1)

        # Interpolate between levels
        r = lerp(c0.r, c1.r, t)
        g = lerp(c0.g, c1.g, t)
        b = lerp(c0.b, c1.b, t)
        a = lerp(c0.a, c1.a, t)

        return Color(r, g, b, a)
    


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
        if self.texture is None:
            return

        self.mipmaps = [self.texture]
        prev = self.texture

        while prev.shape[0] > 1 or prev.shape[1] > 1:
            h_prev, w_prev = prev.shape[0], prev.shape[1]
            h_new = max(1, h_prev // 2)
            w_new = max(1, w_prev // 2)

            new_level = np.zeros((h_new, w_new, 4), dtype=np.uint8)

            for y in range(h_new):
                for x in range(w_new):
                    ys = [min(2*y, h_prev-1), min(2*y+1, h_prev-1)]
                    xs = [min(2*x, w_prev-1), min(2*x+1, w_prev-1)]

                    block = [
                        prev[ys[0], xs[0]],
                        prev[ys[0], xs[1]],
                        prev[ys[1], xs[0]],
                        prev[ys[1], xs[1]],
                    ]
                    block = np.array(block, dtype=np.float32)
                    avg = np.mean(block, axis=0)

                    new_level[y, x] = avg.astype(np.uint8)

            self.mipmaps.append(new_level)
            prev = new_level


def uint8_to_float(value: int) -> float:
    """Convert uint8 color value to float [0, 1] range."""
    return value / 255.0


def float_to_uint8(value: float) -> int:
    """Convert float color value to uint8 [0, 255] range."""
    return int(clamp(value * 255.0, 0, 255))