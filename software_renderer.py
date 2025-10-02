"""
Software Renderer Template - Student Implementation Required

This is the main rendering engine for the 2D SVG rasterizer project.
Students must implement the functions marked with TODO comments.

Tasks Overview:
- Task 0: Line Rasterization (rasterize_line)
- Task 1: Triangle Rasterization (rasterize_triangle)   
- Task 2: Supersampling & Resolve (resolve, fill_sample, fill_pixel) 
- Task 3: Transform Hierarchy (draw_element, set_viewbox, get_matrix) 
- Task 4: Image Rasterization (rasterize_image, texture sampling) 
- Task 5: Alpha Compositing (alpha_blend in math_utils.py) 
"""

import numpy as np
from typing import List, Tuple
from math_utils import Vector2D, Matrix3x3, Color, edge_function
from svg_parser import SVG, SVGElement, SVGPoint, SVGLine, SVGTriangle, SVGPolygon, SVGRect, SVGImage, SVGGroup


class Viewport:
    """Viewport management for pan and zoom operations."""
    
    def __init__(self, x: float, y: float, span: float):
        """Initialize viewport with center and span."""
        self.x = x  # Center x
        self.y = y  # Center y
        self.span = span  # Half of viewport width/height
    
    def set_viewbox(self, x: float, y: float, span: float):
        """Set viewport parameters."""
        self.x = x
        self.y = y
        self.span = span
    
    def update_viewbox(self, dx: float, dy: float, scale: float):
        """Update viewport with delta movement and scale."""
        self.x += dx
        self.y += dy
        self.span *= scale
    
    def get_matrix(self) -> Matrix3x3:
        """Get transformation matrix from viewport space to normalized space."""
        """
        TASK 3: Viewing Transforms - Get Canvas to Normalized Transform
        
        TODO: Create transformation matrix from canvas space to normalized space.
        
        Requirements:
        - Transform canvas region [x-span, x+span] x [y-span, y+span] to [-1,1] x [-1,1]
        - Use viewport parameters (self.x, self.y, self.span)
        - Return proper transformation matrix
        """
        # Translate center (x,y) to origin
        T = Matrix3x3.translation(-self.x, -self.y)
        # Scale so that span maps to 1, range becomes [-1,1]
        S = Matrix3x3.scale(1.0 / self.span, 1.0 / self.span)

        return S * T

class ViewportImp(Viewport):
    """Implementation class for viewport with additional functionality."""
    
    def __init__(self, x: float = 0, y: float = 0, span: float = 1):
        super().__init__(x, y, span)
        self.aspect_ratio = 1.0
    
    def set_aspect_ratio(self, width: float, height: float):
        """Set aspect ratio for viewport."""
        self.aspect_ratio = width / height if height > 0 else 1.0
    
    def get_screen_matrix(self, width: int, height: int) -> Matrix3x3:
        """Get transformation matrix from normalized space to screen space."""
        # Transform from normalized device coordinates [-1, 1] to screen space
        scale_x = width / 2.0
        scale_y = height / 2.0
        tx = width / 2.0
        ty = height / 2.0
        
        return Matrix3x3([
            [scale_x, 0, tx],
            [0, scale_y, ty],
            [0, 0, 1]
        ])


class SoftwareRenderer:
    """Main software renderer class."""
    
    def __init__(self):
        """Initialize renderer."""
        self.viewport = ViewportImp()
        self.sample_rate = 1
        self.width = 0
        self.height = 0
        self.pixel_buffer = None
        self.sample_buffer = None
        self.sample_width = 0
        self.sample_height = 0
        
        # for hierarchical transforms
        self.transform_stack = []
        self.current_transform = Matrix3x3.identity()
    
    def set_pixel_buffer(self, buffer: np.ndarray, width: int, height: int):
        """Set the pixel buffer for rendering."""
        self.pixel_buffer = buffer
        self.width = width
        self.height = height
        
        # Init sample buffer
        self.update_sample_buffer()
    
    def set_sample_rate(self, rate: int):
        """Set supersampling rate."""
        self.sample_rate = max(1, min(8, rate))
        self.update_sample_buffer()
    
    def update_sample_buffer(self):
        """Update sample buffer based on current settings."""
        if self.width > 0 and self.height > 0:
            self.sample_width = self.width * self.sample_rate
            self.sample_height = self.height * self.sample_rate
            
            # sample buffer (RGBA float)
            self.sample_buffer = np.zeros((self.sample_height, self.sample_width, 4), dtype=np.float32)
    
    def clear_buffers(self, color: Color = None):
        """Clear all buffers."""
        if color is None:
            color = Color(1, 1, 1, 1)  # White background
        
        # Clear sample buffer
        if self.sample_buffer is not None:
            self.sample_buffer[:, :] = [color.r, color.g, color.b, color.a]
        
        # Clear pixel buffer
        if self.pixel_buffer is not None:
            r, g, b, a = color.to_rgba_int()
            self.pixel_buffer[:] = 0
            for i in range(0, len(self.pixel_buffer), 4):
                self.pixel_buffer[i] = r
                self.pixel_buffer[i+1] = g
                self.pixel_buffer[i+2] = b
                self.pixel_buffer[i+3] = a
    
    def fill_sample(self, x: int, y: int, color: Color):
        """Fill a single sample in the sample buffer."""
        """
        TASK 2: Fill Sample (Part of Supersampling)
        """
        # Check if coordinates are within bounds
        if 0 <= x < self.sample_width and 0 <= y < self.sample_height:
            # Get existing color from buffer at [y, x]
            existing = self.sample_buffer[y, x]
            existing_color = Color(existing[0], existing[1], existing[2], existing[3])
            
            # Blend new color with existing using alpha blending
            blended_color = color.alpha_blend(existing_color)

            # Store blended result back to buffer
            self.sample_buffer[y, x] = [blended_color.r, blended_color.g, blended_color.b, blended_color.a]
    
    def fill_pixel(self, x: int, y: int, color: Color):
        """Fill a pixel with supersampling."""
        """
        TASK 2: Fill Pixel (Part of Supersampling)
        """
        # For each sample within the pixel (sample_rate x sample_rate)
        for sy in range(self.sample_rate):
            for sx in range(self.sample_rate):
                # Calculate sample coordinates: x*sample_rate + sx, y*sample_rate + sy
                sample_x = x * self.sample_rate + sx
                sample_y = y * self.sample_rate + sy
                
                # Call fill_sample() for each sample coordinate
                self.fill_sample(sample_x, sample_y, color)
    
    def resolve(self):
        """Resolve sample buffer to pixel buffer using box filter."""
        """
        TASK 2: Supersampling & Resolve
        """
        # For each pixel in the output image (width x height)
        for y in range(self.height):
            for x in range(self.width):
                r_sum = g_sum = b_sum = a_sum = 0.0

                # Accumulate all samples for this pixel
                for sy in range(self.sample_rate):
                    for sx in range(self.sample_rate):
                        sx = x * self.sample_rate + sx
                        sy = y * self.sample_rate + sy

                        sample = self.sample_buffer[sy, sx]
                        r_sum += sample[0]
                        g_sum += sample[1]
                        b_sum += sample[2]
                        a_sum += sample[3]

                # Average the accumulated color values
                num_samples = self.sample_rate * self.sample_rate
                avg_r = r_sum / num_samples
                avg_g = g_sum / num_samples
                avg_b = b_sum / num_samples
                avg_a = a_sum / num_samples

                # Write averaged color to pixel_buffer
                idx = (y * self.width + x) * 4
                self.pixel_buffer[idx + 0] = int(avg_r * 255)
                self.pixel_buffer[idx + 1] = int(avg_g * 255)
                self.pixel_buffer[idx + 2] = int(avg_b * 255)
                self.pixel_buffer[idx + 3] = int(avg_a * 255)
    
    def rasterize_point(self, point: Vector2D, color: Color, size: float = 1.0):
        """Rasterize a point."""
        # Transform point to screen space
        p = self.current_transform * point
        
        # Convert to sample coordinates
        screen_x = int(p.x)
        screen_y = int(p.y)
        sample_x = screen_x * self.sample_rate
        sample_y = screen_y * self.sample_rate
        
        # Draw point as small square
        radius = int(size * self.sample_rate)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                self.fill_sample(sample_x + dx, sample_y + dy, color)
    
    def rasterize_line(self, start: Vector2D, end: Vector2D, color: Color, width: float = 1.0):
        """Rasterize a line using Bresenham's algorithm."""
        """
        TASK 0: Line Rasterization
        
        TODO: Implement Bresenham's line algorithm for line drawing.
        
        Requirements:
        - Handle arbitrary slopes and non-integer coordinates
        - Support line width (thickness)
        - Work in sample space coordinates
        - Use fill_sample() to write pixels
        
        Steps to implement:
        1. Transform start and end points using self.current_transform
        2. Convert transformed points to screen space (int coordinates)
        3. Convert screen coordinates to sample space (multiply by sample_rate)
        4. Implement Bresenham's algorithm:
           - Calculate dx, dy, and step directions (sx, sy)
           - Use error accumulation to decide when to step in x vs y
           - Handle line width by drawing multiple pixels around each point
        5. Use fill_sample(x, y, color) to draw each pixel
        
        Hint: The classic Bresenham algorithm uses integer arithmetic and
        an error term to efficiently determine which pixels to draw.
        """
        # Transform start and end points
        p0 = self.current_transform * start
        p1 = self.current_transform * end

        # Convert to screen space
        x0, y0 = int(round(p0.x)), int(round(p0.y))
        x1, y1 = int(round(p1.x)), int(round(p1.y))

        # Convert to sample space
        x0 *= self.sample_rate
        y0 *= self.sample_rate
        x1 *= self.sample_rate
        y1 *= self.sample_rate

        # Bresenham algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            half_w = int(max(1, round(width * self.sample_rate / 2)))
            for i in range(-half_w, half_w + 1):
                for j in range(-half_w, half_w + 1):
                    self.fill_sample(x0 + i, y0 + j, color)

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def rasterize_triangle(self, vertices: List[Vector2D], color: Color):
        """Rasterize a triangle using edge functions."""
        # Transform vertices
        v0 = self.current_transform * vertices[0]
        v1 = self.current_transform * vertices[1]
        v2 = self.current_transform * vertices[2]
        
        # Convert to sample space
        screen_v0 = Vector2D(int(v0.x), int(v0.y))
        screen_v1 = Vector2D(int(v1.x), int(v1.y))
        screen_v2 = Vector2D(int(v2.x), int(v2.y))
        v0_s = Vector2D(screen_v0.x * self.sample_rate, screen_v0.y * self.sample_rate)
        v1_s = Vector2D(screen_v1.x * self.sample_rate, screen_v1.y * self.sample_rate)
        v2_s = Vector2D(screen_v2.x * self.sample_rate, screen_v2.y * self.sample_rate)
        
        # Compute bounding box
        min_x = max(0, int(min(v0_s.x, v1_s.x, v2_s.x)))
        max_x = min(self.sample_width - 1, int(max(v0_s.x, v1_s.x, v2_s.x)) + 1)
        min_y = max(0, int(min(v0_s.y, v1_s.y, v2_s.y)))
        max_y = min(self.sample_height - 1, int(max(v0_s.y, v1_s.y, v2_s.y)) + 1)
        
        # Compute edge function normalizer (2 * triangle area)
        area = edge_function(v0_s, v1_s, v2_s)
        if abs(area) < 1e-6:
            return  # Degenerate triangle
        
        # Rasterize
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = Vector2D(x + 0.5, y + 0.5)  # Sample at pixel center
                
                # Compute edge functions
                w0 = edge_function(v1_s, v2_s, p)
                w1 = edge_function(v2_s, v0_s, p)
                w2 = edge_function(v0_s, v1_s, p)
                
                # Check if inside triangle (all edge functions have same sign as area)
                if area > 0:
                    if w0 >= 0 and w1 >= 0 and w2 >= 0:
                        self.fill_sample(x, y, color)
                else:
                    if w0 <= 0 and w1 <= 0 and w2 <= 0:
                        self.fill_sample(x, y, color)
    
    def triangulate_polygon(self, vertices: List[Vector2D]) -> List[Tuple[int, int, int]]:
        """Triangulate a polygon using improved ear clipping algorithm for non-convex polygons."""
        n = len(vertices)
        if n < 3:
            return []
        if n == 3:
            return [(0, 1, 2)]
        
        triangles = []
        indices = list(range(n))
        
        def sign(p1: Vector2D, p2: Vector2D, p3: Vector2D) -> float:
            """Calculate the sign of the area of a triangle."""
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
        
        def point_in_triangle(pt: Vector2D, v1: Vector2D, v2: Vector2D, v3: Vector2D) -> bool:
            """Check if a point is inside a triangle using sign method."""
            d1 = sign(pt, v1, v2)
            d2 = sign(pt, v2, v3)
            d3 = sign(pt, v3, v1)
            
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            
            return not (has_neg and has_pos)
        
        def is_convex(prev: Vector2D, curr: Vector2D, next: Vector2D) -> bool:
            """Check if the angle at curr is convex (< 180 degrees)."""
            # Cross product to determine if turning left (convex) or right (concave)
            dx1 = curr.x - prev.x
            dy1 = curr.y - prev.y
            dx2 = next.x - curr.x
            dy2 = next.y - curr.y
            return dx1 * dy2 - dy1 * dx2 >= 0
        
        def is_ear(i: int, indices: List[int], vertices: List[Vector2D]) -> bool:
            """Check if vertex i forms an ear in the polygon."""
            n = len(indices)
            if n < 3:
                return False
            
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            prev = indices[prev_idx]
            curr = indices[i]
            next = indices[next_idx]
            
            v_prev = vertices[prev]
            v_curr = vertices[curr]
            v_next = vertices[next]
            
            # Only convex vertices can be ears
            if not is_convex(v_prev, v_curr, v_next):
                return False
            
            # Check if any other vertex is inside this triangle
            for j in range(n):
                if j == prev_idx or j == i or j == next_idx:
                    continue
                
                v_test = vertices[indices[j]]
                
                # Check if this vertex is inside the triangle
                if point_in_triangle(v_test, v_prev, v_curr, v_next):
                    return False
            
            return True
        
        # Main ear clipping loop
        attempts = 0
        max_attempts = n * 2  # Prevent infinite loops
        
        while len(indices) > 3 and attempts < max_attempts:
            ear_found = False
            attempts += 1
            
            # Try to find an ear
            for i in range(len(indices)):
                if is_ear(i, indices, vertices):
                    prev_idx = (i - 1) % len(indices)
                    next_idx = (i + 1) % len(indices)
                    
                    triangles.append((indices[prev_idx], indices[i], indices[next_idx]))
                    indices.pop(i)
                    ear_found = True
                    break
            
            if not ear_found:
                # If no ear found, the polygon might be complex or self-intersecting
                # Use a simple fan triangulation as fallback
                center = indices[0]
                for i in range(1, len(indices) - 1):
                    triangles.append((center, indices[i], indices[i + 1]))
                break
        
        # Add the last triangle
        if len(indices) == 3:
            triangles.append((indices[0], indices[1], indices[2]))
        
        return triangles
    
    def rasterize_polygon(self, vertices: List[Vector2D], color: Color):
        """Rasterize a polygon by triangulation."""
        if len(vertices) < 3:
            return
        
        # Triangulate the polygon
        triangles = self.triangulate_polygon(vertices)
        
        # Rasterize each triangle
        for i0, i1, i2 in triangles:
            self.rasterize_triangle([vertices[i0], vertices[i1], vertices[i2]], color)
    
    def rasterize_rect(self, x: float, y: float, width: float, height: float, color: Color):
        """Rasterize a rectangle."""
        vertices = [
            Vector2D(x, y),
            Vector2D(x + width, y),
            Vector2D(x + width, y + height),
            Vector2D(x, y + height)
        ]
        self.rasterize_polygon(vertices, color)
    
    def rasterize_image(self, img: SVGImage):
        """Rasterize an image with texture sampling."""
        """
        TASK 4: Image Rasterization
        
        TODO: Implement image rasterization with texture sampling.
        
        Requirements:
        - Map screen pixels to texture coordinates (UV mapping)
        - Use trilinear filtering for texture sampling
        - Handle transforms correctly
        - Clamp texture coordinates to [0,1] range
        
        Steps to implement:
        1. Define image corners in local space (x,y) to (x+width, y+height)
        2. Transform corners to screen space using current_transform
        3. Convert to sample space and compute bounding box
        4. Create inverse transform to map screen back to texture space
        5. For each pixel in bounding box:
           - Convert sample position back to world space
           - Calculate UV coordinates: u = (world_x - img.x) / img.width
           - Check if UV is within [0,1] bounds
           - Sample texture using trilinear filtering (with appropriate mip level)
           - Fill sample with sampled color
        """
        # Define corners in local image space
        corners = [
            Vector2D(img.x, img.y),
            Vector2D(img.x + img.width, img.y),
            Vector2D(img.x, img.y + img.height),
            Vector2D(img.x + img.width, img.y + img.height),
        ]

        # Transform corners to screen space
        transformed = [self.current_transform * c for c in corners]

        # Convert to sample space and bounding box
        xs = [int(round(c.x * self.sample_rate)) for c in transformed]
        ys = [int(round(c.y * self.sample_rate)) for c in transformed]

        min_x = max(0, min(xs))
        max_x = min(self.width * self.sample_rate - 1, max(xs))
        min_y = max(0, min(ys))
        max_y = min(self.height * self.sample_rate - 1, max(ys))

        # Build inverse transform: screen → local image space
        local_corners = [
            Vector2D(img.x, img.y),
            Vector2D(img.x + img.width, img.y),
            Vector2D(img.x, img.y + img.height),
        ]
        screen_corners = transformed[:3]
        M_screen = Matrix3x3([
            [screen_corners[1].x - screen_corners[0].x, screen_corners[2].x - screen_corners[0].x, screen_corners[0].x],
            [screen_corners[1].y - screen_corners[0].y, screen_corners[2].y - screen_corners[0].y, screen_corners[0].y],
            [0, 0, 1]
        ])
        M_local = Matrix3x3([
            [img.width, 0, img.x],
            [0, img.height, img.y],
            [0, 0, 1]
        ])
        M = M_screen * M_local.inverse()  # forward transform
        M_inv = M.inverse()  # inverse transform (screen → local)

        # Loop over bounding box
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p_screen = Vector2D(x + 0.5, y + 0.5)
                p_local = M_inv * p_screen

                # Compute UV
                u = (p_local.x - img.x) / img.width
                v = (p_local.y - img.y) / img.height

                if 0 <= u <= 1 and 0 <= v <= 1:
                    color = img.texture.sample_bilinear(u, v)
                    self.fill_sample(x, y, color)
    
    def draw_element(self, element: SVGElement):
        """Draw an SVG element with its transform."""
        """
        TASK 3: Transform Hierarchy - Part 1: Modeling Transforms
        
        TODO: Implement hierarchical transform application for SVG elements.
        
        Requirements:
        - Apply element's local transform to current transform stack
        - Draw element based on its type
        - Properly restore transform stack when done
        
        Steps:
        1. Save current transform by pushing to transform_stack
        2. Multiply current_transform with element's local transform
        3. Draw element based on its type (point, line, triangle, polygon, rect, image, group)
        4. Draw child elements recursively (for groups)
        5. Restore previous transform by popping from stack
        
        Transform hierarchy:
        - Each element can have a local transform (translate, rotate, scale)
        - Transforms accumulate down the hierarchy
        - Child transforms are relative to parent
        """
        # Save current transform
        self.transform_stack.append(self.current_transform)

        # Apply local transform
        if hasattr(element, "transform") and element.transform is not None:
            self.current_transform = self.current_transform * element.transform

        # Draw element based on its type
        if isinstance(element, SVGLine):
            color = element.get_stroke_color()
            width = element.get_stroke_width()
            self.rasterize_line(element.start, element.end, color, width)

        elif isinstance(element, SVGTriangle):
            color = element.get_color()
            self.rasterize_triangle(element.vertices, color)

        elif isinstance(element, SVGPolygon):
            verts = element.vertices
            if len(verts) >= 3:
                color = element.get_color()
                for i in range(1, len(verts) - 1):
                    self.rasterize_triangle([verts[0], verts[i], verts[i+1]], color)

        elif isinstance(element, SVGRect):
            x0, y0 = element.x, element.y
            x1, y1 = x0 + element.width, y0 + element.height
            v0, v1, v2, v3 = Vector2D(x0, y0), Vector2D(x1, y0), Vector2D(x0, y1), Vector2D(x1, y1)

            color = element.get_color()
            self.rasterize_triangle([v0, v1, v2], color)
            self.rasterize_triangle([v2, v1, v3], color)

        elif isinstance(element, SVGImage):
            self.rasterize_image(element)

        elif isinstance(element, SVGGroup):
            for child in element.children:
                self.draw_element(child)

        # Restore previous transform
        self.current_transform = self.transform_stack.pop()
    
    def draw_svg(self, svg: SVG):
        """Draw an entire SVG document."""
        # Clear buffers
        self.clear_buffers()
        
        # Setup viewport transform
        viewport_matrix = self.viewport.get_matrix()
        screen_matrix = self.viewport.get_screen_matrix(self.width, self.height)
        
        # Setup initial transform
        self.current_transform = screen_matrix * viewport_matrix
        
        # Draw all elements
        for element in svg.elements:
            self.draw_element(element)
        
        # Resolve samples to pixels
        self.resolve()


class SoftwareRendererImp(SoftwareRenderer):
    """Implementation class with additional features."""
    
    def __init__(self):
        super().__init__()
        self.enable_antialiasing = True
        self.gamma = 2.2
    
    def gamma_correct(self, color: Color) -> Color:
        """Apply gamma correction."""
        if self.gamma != 1.0:
            inv_gamma = 1.0 / self.gamma
            return Color(
                pow(color.r, inv_gamma),
                pow(color.g, inv_gamma),
                pow(color.b, inv_gamma),
                color.a
            )
        return color
    
    def set_gamma(self, gamma: float):
        """Set gamma value for correction."""
        self.gamma = max(1.0, min(3.0, gamma))