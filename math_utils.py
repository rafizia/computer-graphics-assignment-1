"""
Mathematical Utilities Template - Partial Student Implementation Required

Provides vector, matrix, and color classes for 2D graphics operations.
Most functions are complete, but students must implement alpha blending.

Student Tasks:
- Task 5: Alpha Compositing
  - Color.alpha_blend() - Implement alpha blending for transparency

Complete Implementations Provided:
- Vector2D: 2D vector operations (add, subtract, multiply, normalize, etc.)
- Matrix3x3: 3x3 matrix operations for 2D transforms
- Color: RGBA color representation and conversion utilities
- edge_function(): For triangle rasterization
- Various utility functions (clamp, lerp, etc.)
"""

import math
from typing import Tuple, List, Union


class Vector2D:
    """2D vector class with mathematical operations."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0):
        """Initialize vector with x and y components."""
        self.x = float(x)
        self.y = float(y)
    
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        """Add two vectors."""
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        """Subtract two vectors."""
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2D':
        """Multiply vector by scalar."""
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector2D':
        """Right multiply vector by scalar."""
        return self * scalar
    
    def __truediv__(self, scalar: float) -> 'Vector2D':
        """Divide vector by scalar."""
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def dot(self, other: 'Vector2D') -> float:
        """Compute dot product with another vector."""
        return self.x * other.x + self.y * other.y
    
    def cross(self, other: 'Vector2D') -> float:
        """Compute 2D cross product (returns scalar z-component)."""
        return self.x * other.y - self.y * other.x
    
    def length(self) -> float:
        """Compute vector length."""
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def length_squared(self) -> float:
        """Compute squared vector length."""
        return self.x * self.x + self.y * self.y
    
    def normalize(self) -> 'Vector2D':
        """Return normalized vector."""
        length = self.length()
        if length < 1e-10:
            return Vector2D(0, 0)
        return Vector2D(self.x / length, self.y / length)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)
    
    def __repr__(self) -> str:
        return f"Vector2D({self.x:.3f}, {self.y:.3f})"


class Matrix3x3:
    """3x3 transformation matrix for 2D graphics."""
    
    def __init__(self, data: List[List[float]] = None):
        """Initialize matrix with 3x3 data or identity matrix."""
        if data is None:
            self.data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            self.data = [row[:] for row in data]
    
    @classmethod
    def identity(cls) -> 'Matrix3x3':
        """Create identity matrix."""
        return cls()
    
    @classmethod
    def translation(cls, tx: float, ty: float) -> 'Matrix3x3':
        """Create translation matrix."""
        return cls([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    
    @classmethod
    def rotation(cls, angle_radians: float) -> 'Matrix3x3':
        """Create rotation matrix."""
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)
        return cls([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    
    @classmethod
    def scale(cls, sx: float, sy: float = None) -> 'Matrix3x3':
        """Create scale matrix."""
        if sy is None:
            sy = sx
        return cls([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    
    def __mul__(self, other: Union['Matrix3x3', Vector2D]) -> Union['Matrix3x3', Vector2D]:
        """Multiply matrix with another matrix or transform a vector."""
        if isinstance(other, Matrix3x3):
            result = [[0, 0, 0] for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        result[i][j] += self.data[i][k] * other.data[k][j]
            return Matrix3x3(result)
        elif isinstance(other, Vector2D):
            x = self.data[0][0] * other.x + self.data[0][1] * other.y + self.data[0][2]
            y = self.data[1][0] * other.x + self.data[1][1] * other.y + self.data[1][2]
            w = self.data[2][0] * other.x + self.data[2][1] * other.y + self.data[2][2]
            if abs(w - 1.0) > 1e-10:
                return Vector2D(x / w, y / w)
            return Vector2D(x, y)
        else:
            raise TypeError("Can only multiply with Matrix3x3 or Vector2D")
    
    def inverse(self) -> 'Matrix3x3':
        """Compute matrix inverse."""
        det = (self.data[0][0] * (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1]) -
               self.data[0][1] * (self.data[1][0] * self.data[2][2] - self.data[1][2] * self.data[2][0]) +
               self.data[0][2] * (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0]))
        
        if abs(det) < 1e-10:
            raise ValueError("Matrix is not invertible")
        
        inv = [[0, 0, 0] for _ in range(3)]
        inv[0][0] = (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1]) / det
        inv[0][1] = (self.data[0][2] * self.data[2][1] - self.data[0][1] * self.data[2][2]) / det
        inv[0][2] = (self.data[0][1] * self.data[1][2] - self.data[0][2] * self.data[1][1]) / det
        inv[1][0] = (self.data[1][2] * self.data[2][0] - self.data[1][0] * self.data[2][2]) / det
        inv[1][1] = (self.data[0][0] * self.data[2][2] - self.data[0][2] * self.data[2][0]) / det
        inv[1][2] = (self.data[0][2] * self.data[1][0] - self.data[0][0] * self.data[1][2]) / det
        inv[2][0] = (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0]) / det
        inv[2][1] = (self.data[0][1] * self.data[2][0] - self.data[0][0] * self.data[2][1]) / det
        inv[2][2] = (self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]) / det
        
        return Matrix3x3(inv)


class Color:
    """RGBA color class with blending operations."""
    
    def __init__(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0):
        """Initialize color with RGBA components (0.0-1.0 range)."""
        self.r = max(0.0, min(1.0, float(r)))
        self.g = max(0.0, min(1.0, float(g)))
        self.b = max(0.0, min(1.0, float(b)))
        self.a = max(0.0, min(1.0, float(a)))
    
    @classmethod
    def from_hex(cls, hex_color: str) -> 'Color':
        """Create color from hex string."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return cls(r, g, b, 1.0)
        elif len(hex_color) == 8:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            a = int(hex_color[6:8], 16) / 255.0
            return cls(r, g, b, a)
        else:
            raise ValueError("Invalid hex color format")
    
    @classmethod
    def from_rgba_int(cls, r: int, g: int, b: int, a: int = 255) -> 'Color':
        """Create color from integer RGBA values (0-255 range)."""
        return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    
    def to_rgba_int(self) -> Tuple[int, int, int, int]:
        """Convert to integer RGBA tuple (0-255 range)."""
        return (
            int(self.r * 255),
            int(self.g * 255),
            int(self.b * 255),
            int(self.a * 255)
        )
    
    def alpha_blend(self, background: 'Color') -> 'Color':
        """Alpha blend this color over background color."""
        """
        TASK 5: Alpha Compositing
        
        TODO: Implement alpha blending for transparency.
        
        Requirements:
        - Implement standard alpha blending formula
        - Handle both color and alpha channels correctly
        - Self (foreground) color blends over background color
        
        Formula:
        - result_rgb = alpha * foreground_rgb + (1 - alpha) * background_rgb
        - result_alpha = alpha + (1 - alpha) * background_alpha
        
        Steps:
        1. Extract alpha from foreground (self.a)
        2. Calculate inverse alpha (1 - alpha)
        3. Blend each RGB channel using the formula
        4. Blend alpha channel
        5. Return new Color with blended values
        """

        pass  # Remove this line when implementing
    
    def __mul__(self, factor: float) -> 'Color':
        """Multiply color by factor."""
        return Color(self.r * factor, self.g * factor, self.b * factor, self.a)
    
    def __add__(self, other: 'Color') -> 'Color':
        """Add two colors."""
        return Color(
            min(1.0, self.r + other.r),
            min(1.0, self.g + other.g),
            min(1.0, self.b + other.b),
            min(1.0, self.a + other.a)
        )
    
    def __repr__(self) -> str:
        return f"Color({self.r:.3f}, {self.g:.3f}, {self.b:.3f}, {self.a:.3f})"


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def barycentric_coordinates(p: Vector2D, a: Vector2D, b: Vector2D, c: Vector2D) -> Tuple[float, float, float]:
    """Compute barycentric coordinates of point p with respect to triangle abc."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    
    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)
    
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v
    
    return w, v, u  # Return in order (alpha, beta, gamma) for vertices (a, b, c)


def edge_function(a: Vector2D, b: Vector2D, c: Vector2D) -> float:
    """Compute edge function for triangle rasterization."""
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)