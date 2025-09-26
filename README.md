# SVG Rasterizer - Student Template

## Overview
A Python implementation of a 2D SVG software rasterizer for Computer Graphics education. Students will implement core rasterization algorithms to render SVG graphics.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Run with single SVG file
python draw_svg.py svg/basic/test1.svg

# Run with directory (loads all SVG files)
python draw_svg.py svg/basic

# Run without arguments (use GUI file dialogs)
python draw_svg.py
```

## Controls
- **Mouse drag**: Pan viewport
- **Mouse wheel**: Zoom in/out
- **Prev/Next buttons**: Navigate between SVG files
- **1/2/3 buttons**: Set sample rate for anti-aliasing
- **Reset View**: Reset viewport to fit entire SVG
- **Show Reference**: Toggle reference mode (if available)

## Student Tasks

Your implementation must complete these core functions:

### Task 0: Line Rasterization
- `rasterize_line()` in `software_renderer.py`

### Task 1: Triangle Rasterization 
- `rasterize_triangle()` in `software_renderer.py`

### Task 2: Supersampling & Anti-aliasing 
- `fill_sample()` in `software_renderer.py`
- `fill_pixel()` in `software_renderer.py`
- `resolve()` in `software_renderer.py`

### Task 3: Transform Hierarchy 
- `get_matrix()` in `software_renderer.py`
- `draw_element()` in `software_renderer.py`

### Task 4: Image Rasterization 
- `rasterize_image()` in `software_renderer.py`
- `sample_nearest()` in `texture.py`
- `sample_bilinear()` in `texture.py`
- `sample_trilinear()` in `texture.py`
- `generate_mips()` in `texture.py`

### Task 5: Alpha Compositing 
- `alpha_blend()` in `math_utils.py`

## Testing
Start with simple SVG files and progress to more complex ones:
1. Basic geometric shapes
2. Complex polygons and self-intersecting shapes
3. Transform hierarchies and coordinate systems
4. Transparency and alpha blending
5. Image textures and filtering

## File Structure
- `draw_svg.py` - Main GUI application
- `software_renderer.py` - Core rendering engine (main implementation file)
- `svg_parser.py` - SVG file parser
- `math_utils.py` - Mathematical utilities (Vector2D, Matrix3x3, Color)
- `texture.py` - Texture sampling utilities
- `svg/` - Test SVG files organized by complexity
