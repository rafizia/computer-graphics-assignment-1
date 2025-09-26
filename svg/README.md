# SVG Test Files

This directory contains test SVG files organized by feature for testing the Python SVG rasterizer implementation.

## Directory Structure

### `basic/` - Basic Shape Tests
- `test1.svg` - Simple points (circles) test
- `test2.svg` - Line drawing with various slopes
- `test3.svg` - Triangle rasterization test
- `test4.svg` - Rectangle rendering test
- `test5.svg` - Complex polygon shapes
- `test6.svg` - Transform hierarchy test
- `test7.svg` - Nested groups with transforms
- `flower.svg` - Flower pattern made of points (similar to Stanford reference)
- `complex_scene.svg` - Complete scene with all basic elements

### `alpha/` - Transparency Tests
- `test1.svg` - Basic alpha blending with overlapping rectangles
- `test2.svg` - Complex multi-layer alpha compositing
- `test3.svg` - Varying opacity levels

### `debug/` - Simple Debug Tests
- `simple_line.svg` - Minimal line for debugging Task 0
- `simple_triangle.svg` - Minimal triangle for debugging Task 1
- `simple_rect.svg` - Minimal rectangle for debugging

### `image/` - Image Rendering Tests
- `test1.svg` - Placeholder for image tests (requires actual image files)

## Usage

Run tests with the viewer:
```bash
# Test specific file
python draw_svg.py svg/basic/test1.svg

# Test entire directory
python draw_svg.py svg/basic

# Debug specific issues
python draw_svg.py svg/debug/simple_line.svg
```

## Testing Strategy

1. **Start with debug files** - Use simple shapes to verify basic functionality
2. **Progress through basic tests** - Test each feature incrementally
3. **Verify transforms** - Test hierarchical transforms and viewport
4. **Check alpha blending** - Verify transparency implementation
5. **Complex scenes** - Test complete rendering pipeline

## Expected Behavior

### Task 0 (Lines)
- `test2.svg` should show lines of various slopes and orientations
- Lines should be connected and have proper thickness

### Task 1 (Triangles) 
- `test3.svg` should show filled triangles with proper coverage
- No gaps or holes in triangle fills
- Correct handling of winding order

### Task 2 (Supersampling)
- Increase sample rate with `+` key to see smoother edges
- Anti-aliasing should reduce jagged edges on diagonal lines

### Task 3 (Transforms)
- `test6.svg` and `test7.svg` should show proper hierarchical transforms
- Mouse pan/zoom should work correctly

### Task 4 (Images)
- Requires placing actual image files and updating SVG references
- Should show bilinear filtered scaling

### Task 5 (Alpha)
- `alpha/` directory tests should show proper color blending
- Semi-transparent overlays should composite correctly

## Creating Additional Tests

To create new test files:
1. Keep SVG structure simple - avoid complex paths
2. Use supported elements: rect, circle, line, polygon
3. Test edge cases like degenerate triangles
4. Include both clockwise and counter-clockwise winding
5. Test transform edge cases (zero scale, large translations)

## Debugging Tips

- Use `debug/` files to isolate specific issues
- Check console output for error messages
- Use pixel inspector mode (if implemented) for detailed analysis
- Compare with reference implementation when available