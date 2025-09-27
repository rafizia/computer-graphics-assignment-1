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

        # Clamp UV dan pilih mipmap level
        u = clamp(u, 0.0, 1.0)
        v = clamp(v, 0.0, 1.0)
        level = clamp(level, 0, len(self.mipmaps) - 1)
        mip_texture = self.mipmaps[level]
        
        mip_height, mip_width = mip_texture.shape[:2]
        
        # Mengonversi UV ke pixel coordinates dan round ke nearest
        x = u * (mip_width - 1)
        y = v * (mip_height - 1)
        px = int(round(x))
        py = int(round(y))
        
        px = clamp(px, 0, mip_width - 1)
        py = clamp(py, 0, mip_height - 1)
        
        # Sample pixel dan mengonversi ke Color
        pixel = mip_texture[py, px]

        if pixel.dtype == np.uint8:
            if len(pixel) == 4:  # RGBA
                return Color(pixel[0]/255.0, pixel[1]/255.0, pixel[2]/255.0, pixel[3]/255.0)
            elif len(pixel) == 3:  # RGB
                return Color(pixel[0]/255.0, pixel[1]/255.0, pixel[2]/255.0, 1.0)
        else: # Kalo udah float
            if len(pixel) == 4:  # RGBA
                return Color(pixel[0], pixel[1], pixel[2], pixel[3])
            elif len(pixel) == 3:  # RGB
                return Color(pixel[0], pixel[1], pixel[2], 1.0)
        
        return Color(0.0, 0.0, 0.0, 1.0)
    
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
        
        # Clamp UV dan pilih mipmap level
        u = clamp(u, 0.0, 1.0)
        v = clamp(v, 0.0, 1.0)
        level = clamp(level, 0, len(self.mipmaps) - 1)
        mip_texture = self.mipmaps[level]
        
        mip_height, mip_width = mip_texture.shape[:2]
        
         # Mengonversi UV ke pixel coordinates dan ambil integer + fractional parts
        x = u * (mip_width - 1)
        y = v * (mip_height - 1)
        x_int = int(x)
        y_int = int(y)
        fx = x - x_int
        fy = y - y_int
        
        # Clamp coordinates untuk 4 surrounding pixels
        x0 = clamp(x_int, 0, mip_width - 1)
        x1 = clamp(x_int + 1, 0, mip_width - 1)
        y0 = clamp(y_int, 0, mip_height - 1)
        y1 = clamp(y_int + 1, 0, mip_height - 1)
        
        def get_pixel_color(py, px):
            pixel = mip_texture[py, px]
            if pixel.dtype == np.uint8:
                if len(pixel) >= 3:
                    r, g, b = pixel[0]/255.0, pixel[1]/255.0, pixel[2]/255.0
                    a = pixel[3]/255.0 if len(pixel) == 4 else 1.0
                    return [r, g, b, a]
            else:
                if len(pixel) >= 3:
                    r, g, b = pixel[0], pixel[1], pixel[2]
                    a = pixel[3] if len(pixel) == 4 else 1.0
                    return [r, g, b, a]
            return [0.0, 0.0, 0.0, 1.0]
        
        p00 = get_pixel_color(y0, x0)
        p10 = get_pixel_color(y0, x1)
        p01 = get_pixel_color(y1, x0)
        p11 = get_pixel_color(y1, x1)
        
        # Bilinear interpolation: horizontal dulu, terus vertical
        top = [(1-fx) * p00[i] + fx * p10[i] for i in range(4)]
        bottom = [(1-fx) * p01[i] + fx * p11[i] for i in range(4)]
        final = [(1-fy) * top[i] + fy * bottom[i] for i in range(4)]
        
        return Color(final[0], final[1], final[2], final[3])
    
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

        # Clamp mip level dan ambil integer + fractional parts
        max_level = len(self.mipmaps) - 1
        mip_level = clamp(mip_level, 0.0, float(max_level))
        level0 = int(mip_level)
        level1 = min(level0 + 1, max_level)
        t = mip_level - level0
        
        # Kalo fractional kecil atau di level tertinggi, pakai bilinear aja
        if t < 1e-6 or level0 == max_level:
            return self.sample_bilinear(u, v, level0)
        
        # Sample dari 2 mip levels dan interpolasi
        color0 = self.sample_bilinear(u, v, level0)
        color1 = self.sample_bilinear(u, v, level1)
        
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
        
        # Meng-generate level-level yang progressively smaller
        height, width = self.texture.shape[:2]
        if width <= 0 or height <= 0:
            return
        
        self.mipmaps = [self.texture]
        current_level = self.texture
        
        # Meng-generate level" yang progressively smaller
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