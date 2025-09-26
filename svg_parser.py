"""
SVG parsing and data structures for the 2D rasterizer.
Handles parsing of SVG files and creating appropriate data structures.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import math
import os
import json
import numpy as np
from math_utils import Vector2D, Matrix3x3, Color
from PIL import Image
from texture import Sampler2DImp

class SVGElement:
    """Base class for SVG elements."""
    
    # Load CSS color names once as class variable
    _css_colors = None
    
    @classmethod
    def _load_css_colors(cls):
        """Load CSS color names from JSON file."""
        if cls._css_colors is None:
            try:
                css_colors_path = os.path.join(os.path.dirname(__file__), 'css-color-names.json')
                with open(css_colors_path, 'r') as f:
                    cls._css_colors = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load CSS colors: {e}")
                cls._css_colors = {}
        return cls._css_colors
    
    def __init__(self, element_type: str):
        self.element_type = element_type
        self.transform = Matrix3x3.identity()
        self.style = {}
        self.attributes = {}
        self._load_css_colors()
    
    def set_transform(self, transform: Matrix3x3):
        """Set transformation matrix for this element."""
        self.transform = transform
    
    def set_style(self, style: Dict[str, Any]):
        """Set style properties for this element."""
        self.style = style
    
    def get_color(self) -> Color:
        """Get fill color from style."""
        fill = self.style.get('fill', '#000000')
        opacity = float(self.style.get('opacity', 1.0))
        fill_opacity = float(self.style.get('fill-opacity', 1.0))
        
        if fill == 'none':
            return Color(0, 0, 0, 0)
        
        if fill.startswith('#'):
            color = Color.from_hex(fill)
        else:
            # Handle named colors using CSS color names
            css_colors = self._load_css_colors()
            hex_color = css_colors.get(fill.lower())
            if hex_color:
                color = Color.from_hex(hex_color)
            else:
                # fallback
                color = Color(0, 0, 0)
        
        color.a *= opacity * fill_opacity
        return color
    
    def get_stroke_color(self) -> Color:
        """Get stroke color from style."""
        stroke = self.style.get('stroke', 'none')
        opacity = float(self.style.get('opacity', 1.0))
        stroke_opacity = float(self.style.get('stroke-opacity', 1.0))
        
        if stroke == 'none':
            return Color(0, 0, 0, 0)  # Transparent
        
        if stroke.startswith('#'):
            color = Color.from_hex(stroke)
        else:
            # Handle named colors using CSS color names
            css_colors = self._load_css_colors()
            hex_color = css_colors.get(stroke.lower())
            if hex_color:
                color = Color.from_hex(hex_color)
            else:
                # fallback
                color = Color(0, 0, 0)
        
        color.a *= opacity * stroke_opacity
        return color
    
    def get_stroke_width(self) -> float:
        """Get stroke width from style."""
        return float(self.style.get('stroke-width', 1.0))


class SVGPoint(SVGElement):
    """SVG point element."""
    
    def __init__(self, position: Vector2D):
        super().__init__('point')
        self.position = position


class SVGLine(SVGElement):
    """SVG line element."""
    
    def __init__(self, start: Vector2D, end: Vector2D):
        super().__init__('line')
        self.start = start
        self.end = end


class SVGTriangle(SVGElement):
    """SVG triangle element."""
    
    def __init__(self, vertices: List[Vector2D]):
        super().__init__('triangle')
        if len(vertices) != 3:
            raise ValueError("Triangle must have exactly 3 vertices")
        self.vertices = vertices


class SVGPolygon(SVGElement):
    """SVG polygon element."""
    
    def __init__(self, vertices: List[Vector2D]):
        super().__init__('polygon')
        self.vertices = vertices


class SVGRect(SVGElement):
    """SVG rectangle element."""
    
    def __init__(self, x: float, y: float, width: float, height: float):
        super().__init__('rect')
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class SVGCircle(SVGElement):
    """SVG circle element."""
    
    def __init__(self, cx: float, cy: float, r: float):
        super().__init__('circle')
        self.cx = cx
        self.cy = cy
        self.r = r
    
    def to_polygon(self, num_segments: int = 32) -> SVGPolygon:
        """Convert circle to polygon approximation."""
        vertices = []
        for i in range(num_segments):
            angle = 2 * math.pi * i / num_segments
            x = self.cx + self.r * math.cos(angle)
            y = self.cy + self.r * math.sin(angle)
            vertices.append(Vector2D(x, y))
        
        poly = SVGPolygon(vertices)
        poly.style = self.style.copy()
        poly.transform = self.transform
        return poly


class Texture:
    """Texture class for image sampling."""
    
    def __init__(self, width: int, height: int, data: np.ndarray = None):
        """Initialize texture with dimensions and optional data."""
        self.width = width
        self.height = height
        
        if data is None:
            # empty texture
            self.data = np.zeros((height, width, 4), dtype=np.uint8)
        else:
            self.data = data
        
        # sampler
        self.sampler = Sampler2DImp(self.data)
        self.sampler.generate_mips()
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Texture':
        """Load texture from image file."""
        if Image is None:
            raise ImportError("PIL/Pillow is required to load image files")
        
        if not os.path.exists(filepath):
            # Return checkerboard texture if file not found
            return cls.create_checkerboard(256, 256)
        
        try:
            img = Image.open(filepath)
            img = img.convert('RGBA')
            
            data = np.array(img, dtype=np.uint8)
            
            return cls(img.width, img.height, data)
        except Exception as e:
            print(f"Error loading texture {filepath}: {e}")
            # Return checkerboard texture on error
            return cls.create_checkerboard(256, 256)
    
    @classmethod
    def create_checkerboard(cls, width: int, height: int, size: int = 16) -> 'Texture':
        """Create a checkerboard pattern texture."""
        data = np.zeros((height, width, 4), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                checker = ((x // size) + (y // size)) % 2
                if checker:
                    data[y, x] = [255, 255, 255, 255] 
                else:
                    data[y, x] = [200, 200, 200, 255] 
        
        return cls(width, height, data)


class SVGImage(SVGElement):
    """SVG image element."""
    
    def __init__(self, x: float, y: float, width: float, height: float, texture: Texture):
        super().__init__('image')
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.texture = texture


class SVGGroup(SVGElement):
    """SVG group element containing other elements."""
    
    def __init__(self):
        super().__init__('group')
        self.children = []
    
    def add_child(self, element: SVGElement):
        """Add child element to group."""
        self.children.append(element)


class SVG:
    """Main SVG document class."""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.elements = []
        self.viewbox = (0, 0, width, height)
    
    def add_element(self, element: SVGElement):
        """Add element to SVG document."""
        self.elements.append(element)
    
    def set_viewbox(self, x: float, y: float, width: float, height: float):
        """Set SVG viewbox."""
        self.viewbox = (x, y, width, height)


class SVGParser:
    """Parser for SVG files."""
    
    def __init__(self):
        self.current_transform = Matrix3x3.identity()
        self.transform_stack = []
        self.style_stack = []
        self.current_style = {}
    
    def parse_file(self, filepath: str) -> SVG:
        """Parse SVG file and return SVG object."""
        tree = ET.parse(filepath)
        root = tree.getroot()
        return self.parse_svg(root, os.path.dirname(filepath))
    
    def parse_svg(self, root: ET.Element, base_dir: str = "") -> SVG:
        """Parse SVG root element."""
        # Get dimensions
        width = self.parse_length(root.get('width', '100'))
        height = self.parse_length(root.get('height', '100'))
        
        svg = SVG(width, height)
        
        # Parse viewBox if present
        viewbox_str = root.get('viewBox')
        if viewbox_str:
            parts = viewbox_str.split()
            if len(parts) == 4:
                vb_x = float(parts[0])
                vb_y = float(parts[1])
                vb_width = float(parts[2])
                vb_height = float(parts[3])
                svg.set_viewbox(vb_x, vb_y, vb_width, vb_height)
                
                # Apply viewbox transform
                scale_x = width / vb_width
                scale_y = height / vb_height
                translate = Matrix3x3.translation(-vb_x * scale_x, -vb_y * scale_y)
                scale = Matrix3x3.scale(scale_x, scale_y)
                self.current_transform = scale * translate
        
        # Parse children
        for child in root:
            element = self.parse_element(child, base_dir)
            if element:
                svg.add_element(element)
        
        return svg
    
    def parse_element(self, node: ET.Element, base_dir: str = "") -> Optional[SVGElement]:
        """Parse individual SVG element."""
        # Remove namespace if present
        tag = node.tag.split('}')[-1] if '}' in node.tag else node.tag
        
        # Parse style and transform
        style = self.parse_style(node)
        transform = self.parse_transform(node.get('transform', ''))
        
        element = None
        
        if tag == 'rect':
            x = self.parse_length(node.get('x', '0'))
            y = self.parse_length(node.get('y', '0'))
            width = self.parse_length(node.get('width', '0'))
            height = self.parse_length(node.get('height', '0'))
            element = SVGRect(x, y, width, height)
        
        elif tag == 'circle':
            cx = self.parse_length(node.get('cx', '0'))
            cy = self.parse_length(node.get('cy', '0'))
            r = self.parse_length(node.get('r', '0'))
            circle = SVGCircle(cx, cy, r)
            # Convert to polygon for rasterization
            element = circle.to_polygon(32)
            element.set_style(self.current_style.copy())
            element.style.update(style)
            element.set_transform(transform * self.current_transform)
        
        elif tag == 'polygon':
            points_str = node.get('points', '')
            vertices = self.parse_points(points_str)
            if vertices:
                element = SVGPolygon(vertices)
        
        elif tag == 'line':
            x1 = self.parse_length(node.get('x1', '0'))
            y1 = self.parse_length(node.get('y1', '0'))
            x2 = self.parse_length(node.get('x2', '0'))
            y2 = self.parse_length(node.get('y2', '0'))
            element = SVGLine(Vector2D(x1, y1), Vector2D(x2, y2))
        
        elif tag == 'g':
            # Group element
            group = SVGGroup()
            
            # Push current state
            self.transform_stack.append(self.current_transform)
            self.style_stack.append(self.current_style.copy())
            
            # Update state
            self.current_transform = transform * self.current_transform
            self.current_style.update(style)
            
            # Parse children
            for child in node:
                child_element = self.parse_element(child, base_dir)
                if child_element:
                    group.add_child(child_element)
            
            # Pop state
            self.current_transform = self.transform_stack.pop()
            self.current_style = self.style_stack.pop()
            
            element = group
        
        elif tag == 'image':
            x = self.parse_length(node.get('x', '0'))
            y = self.parse_length(node.get('y', '0'))
            width = self.parse_length(node.get('width', '0'))
            height = self.parse_length(node.get('height', '0'))
            
            # Load texture
            href = node.get('{http://www.w3.org/1999/xlink}href') or node.get('href', '')
            if href:
                if href.startswith('data:'):
                    # Data URL - create checkerboard for now
                    texture = Texture.create_checkerboard(256, 256)
                else:
                    # File path
                    filepath = os.path.join(base_dir, href)
                    texture = Texture.from_file(filepath)
            else:
                texture = Texture.create_checkerboard(256, 256)
            
            element = SVGImage(x, y, width, height, texture)
        
        if element:
            element.set_style(self.current_style.copy())
            element.style.update(style)
            element.set_transform(transform)
        
        return element
    
    def parse_style(self, node: ET.Element) -> Dict[str, str]:
        """Parse style attributes from node."""
        style = {}
        
        # Direct attributes
        for attr in ['fill', 'stroke', 'stroke-width', 'opacity', 'fill-opacity', 'stroke-opacity']:
            value = node.get(attr)
            if value:
                style[attr] = value
        
        # Style attribute
        style_str = node.get('style', '')
        if style_str:
            for item in style_str.split(';'):
                if ':' in item:
                    key, value = item.split(':', 1)
                    style[key.strip()] = value.strip()
        
        return style
    
    def parse_transform(self, transform_str: str) -> Matrix3x3:
        """Parse transform string and return transformation matrix."""
        if not transform_str:
            return Matrix3x3.identity()
        
        result = Matrix3x3.identity()
        
        # Simple parsing for basic transforms
        transforms = transform_str.strip().split(')')
        
        for transform in transforms:
            if not transform:
                continue
            
            transform = transform.strip()
            if '(' not in transform:
                continue
            
            func_name, params_str = transform.split('(', 1)
            func_name = func_name.strip()
            params = [float(p) for p in params_str.replace(',', ' ').split()]
            
            if func_name == 'translate':
                if len(params) == 1:
                    result = Matrix3x3.translation(params[0], 0) * result
                elif len(params) >= 2:
                    result = Matrix3x3.translation(params[0], params[1]) * result
            
            elif func_name == 'scale':
                if len(params) == 1:
                    result = Matrix3x3.scale(params[0], params[0]) * result
                elif len(params) >= 2:
                    result = Matrix3x3.scale(params[0], params[1]) * result
            
            elif func_name == 'rotate':
                if len(params) == 1:
                    # Rotate around origin
                    angle = math.radians(params[0])
                    result = Matrix3x3.rotation(angle) * result
                elif len(params) == 3:
                    # Rotate around point
                    angle = math.radians(params[0])
                    cx, cy = params[1], params[2]
                    t1 = Matrix3x3.translation(cx, cy)
                    r = Matrix3x3.rotation(angle)
                    t2 = Matrix3x3.translation(-cx, -cy)
                    result = (t1 * r * t2) * result
            
            elif func_name == 'matrix':
                if len(params) == 6:
                    # SVG matrix order: a, b, c, d, e, f
                    # Maps to: [[a, c, e], [b, d, f], [0, 0, 1]]
                    mat = Matrix3x3([
                        [params[0], params[2], params[4]],
                        [params[1], params[3], params[5]],
                        [0, 0, 1]
                    ])
                    result = mat * result
        
        return result
    
    def parse_points(self, points_str: str) -> List[Vector2D]:
        """Parse points string for polygons."""
        vertices = []
        
        if not points_str:
            return vertices
        
        coords = points_str.replace(',', ' ').split()
        
        # Parse pairs of coordinates
        for i in range(0, len(coords) - 1, 2):
            try:
                x = float(coords[i])
                y = float(coords[i + 1])
                vertices.append(Vector2D(x, y))
            except (ValueError, IndexError):
                break
        
        return vertices
    
    def parse_length(self, value: str) -> float:
        """Parse length value (handle units)."""
        if not value:
            return 0.0
        
        value = value.strip()
        
        # Remove units
        for unit in ['px', 'pt', 'em', '%']:
            if value.endswith(unit):
                value = value[:-len(unit)]
                break
        
        try:
            return float(value)
        except ValueError:
            return 0.0