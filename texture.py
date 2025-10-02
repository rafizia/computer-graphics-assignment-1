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
        """
        # Clamp u,v to [0,1] range using clamp() function
        u = clamp(u, 0.0, 1.0)
        v = clamp(v, 0.0, 1.0)

        # Select mipmap level
        level = int(clamp(level, 0, len(self.mipmaps) - 1))
        mip_texture = self.mipmaps[level]
        h, w = mip_texture.shape[0], mip_texture.shape[1]

        # Convert UV to pixel coordinates
        x = int(round(u * (w - 1)))
        y = int(round(v * (h - 1)))

        # Clamp pixel coordinates to texture bounds
        x = clamp(x, 0, w - 1)
        y = clamp(y, 0, h - 1)

        # Sample pixel (RGBA in [0,255])
        r, g, b, a = mip_texture[y, x]

        # Convert to Color (float 0-1)
        return Color(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    
    def sample_bilinear(self, u: float, v: float, level: int = 0) -> Color:
        """Sample texture using bilinear filtering."""
        """
        TASK 4: Image Rasterization - Bilinear Filtering
        """ 
        # Clamp UV coordinates to [0,1] range
        u = clamp(u, 0.0, 1.0)
        v = clamp(v, 0.0, 1.0)

        # Clamp mip level
        level = int(clamp(level, 0, len(self.mipmaps) - 1))
        mip_texture = self.mipmaps[level]
        h, w = mip_texture.shape[0], mip_texture.shape[1]

        # Convert UV to pixel coordinates
        x = u * (w - 1)
        y = v * (h - 1)

        # Get integer and fractional parts
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        fx = x - x0
        fy = y - y0

        # Sample four pixels
        p00 = mip_texture[y0, x0]  # top-left
        p10 = mip_texture[y0, x1]  # top-right
        p01 = mip_texture[y1, x0]  # bottom-left
        p11 = mip_texture[y1, x1]  # bottom-right

        # Convert to float
        p00 = [c / 255.0 for c in p00]
        p10 = [c / 255.0 for c in p10]
        p01 = [c / 255.0 for c in p01]
        p11 = [c / 255.0 for c in p11]

        # Perform bilinear interpolation
        top = [(1-fx) * p00[i] + fx * p10[i] for i in range(4)]
        bottom = [(1-fx) * p01[i] + fx * p11[i] for i in range(4)]
        final = [(1-fy) * top[i] + fy * bottom[i] for i in range(4)]

        # Return interpolated color
        return Color(*final)
    
    def sample_trilinear(self, u: float, v: float, mip_level: float) -> Color:
        """Sample texture using trilinear filtering."""
        """
        TASK 4: Trilinear Filtering
        """
        # Clamp mip_level to valid range + get integer and fractional parts
        max_level = len(self.mipmaps) - 1
        mip_level = clamp(mip_level, 0.0, float(max_level))
        level0 = int(mip_level)
        level1 = min(level0 + 1, max_level)
        t = mip_level - level0
        
        # If fractional part is near 0 or at highest level, use bilinear only
        if t < 1e-6 or level0 == max_level:
            return self.sample_bilinear(u, v, level0)
        
        # Sample from two adjacent mip levels using bilinear sampling
        color0 = self.sample_bilinear(u, v, level0)
        color1 = self.sample_bilinear(u, v, level1)
        
        # Interpolate between the two colors using linear interpolation (lerp)
        result_r = lerp(color0.r, color1.r, t)
        result_g = lerp(color0.g, color1.g, t)
        result_b = lerp(color0.b, color1.b, t)
        result_a = lerp(color0.a, color1.a, t)
        
        return Color(result_r, result_g, result_b, result_a)
    


class Sampler2DImp(Sampler2D):
    """Implementation class for Sampler2D with mipmap generation."""
    
    def generate_mips(self):
        """Generate mipmap hierarchy for the texture."""
        """
        TASK 4: Mipmap Generation (Part of Trilinear Filtering)
        """
        if self.texture is None:
            return
        
        # Meng-generate level-level yang progressively smaller
        height, width = self.texture.shape[:2]
        if width <= 0 or height <= 0:
            return
        
        self.mipmaps = [self.texture]
        current_level = self.texture
        
        # Meng-generate level-level yang progressively smaller
        while True:
            current_height, current_width = current_level.shape[:2]
            new_width = max(1, current_width // 2)
            new_height = max(1, current_height // 2)
            
            if new_width == current_width and new_height == current_height:
                break
            
            # Bikin mipmap array baru
            if len(current_level.shape) == 3:  # Kalo berwarna
                channels = current_level.shape[2]
                new_level = np.zeros((new_height, new_width, channels), dtype=current_level.dtype)
            else:  # Grayscale
                new_level = np.zeros((new_height, new_width), dtype=current_level.dtype)
            
            # Box filter: rata-rata 2x2 block untuk setiap pixel baru
            for y in range(new_height):
                for x in range(new_width):
                    src_x = x * 2
                    src_y = y * 2
                    
                    samples = []
                    for dy in range(2):
                        for dx in range(2):
                            sample_x = min(src_x + dx, current_width - 1)
                            sample_y = min(src_y + dy, current_height - 1)
                            samples.append(current_level[sample_y, sample_x])
                    
                    if len(samples) > 0:
                        avg_sample = np.mean(samples, axis=0)
                        new_level[y, x] = avg_sample.astype(current_level.dtype)
            
            self.mipmaps.append(new_level)
            current_level = new_level


def uint8_to_float(value: int) -> float:
    """Convert uint8 color value to float [0, 1] range."""
    return value / 255.0


def float_to_uint8(value: float) -> int:
    """Convert float color value to uint8 [0, 255] range."""
    return int(clamp(value * 255.0, 0, 255))