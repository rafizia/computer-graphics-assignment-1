import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import argparse
from software_renderer import SoftwareRendererImp
from svg_parser import SVGParser


class SVGViewerTkinter:
    """Tkinter-based SVG viewer application."""
    
    def __init__(self, width: int = 800, height: int = 600):
        """Initialize the viewer."""
        self.width = width
        self.height = height
        
        # Init renderer
        self.renderer = SoftwareRendererImp()
        self.pixel_buffer = np.zeros((height * width * 4,), dtype=np.uint8)
        self.renderer.set_pixel_buffer(self.pixel_buffer, width, height)
        
        # SVG management
        self.svg_files = []
        self.current_svg_index = 0
        self.current_svg = None
        
        # Application state
        self.show_reference = False
        self.sample_rate = 1
        self.canvas_scale = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        
        self.setup_gui()
        
        print("SVG Viewer (Tkinter) Controls:")
        print("- Use Prev/Next buttons to navigate between SVG files")
        print("- Click and drag canvas to pan")
        print("- Mouse wheel to zoom in/out")
        print("- Use 1/2/3 buttons to set sample rate")
        print("- Toggle Reference checkbox to compare implementations")
    
    def setup_gui(self):
        """Setup the tkinter GUI."""
        self.root = tk.Tk()
        self.root.title("DrawSVG - Tkinter Implementation")
        self.root.geometry(f"{self.width + 300}x{self.height + 100}")
        
        # main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # canvas frame
        canvas_frame = ttk.LabelFrame(main_frame, text="SVG Display")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # canvas for SVG display
        self.canvas = tk.Canvas(canvas_frame, width=self.width, height=self.height, 
                               bg='white', relief=tk.SUNKEN, bd=2)
        self.canvas.pack(padx=5, pady=5)
        
        # bind mouse events to canvas
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        
        # control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # file selection section
        file_frame = ttk.LabelFrame(control_frame, text="File Selection")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load SVG File", command=self.load_file).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load SVG Directory", command=self.load_directory).pack(fill=tk.X, pady=2)
        
        # navigation section
        nav_frame = ttk.LabelFrame(control_frame, text="Navigation")
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        nav_buttons_frame = ttk.Frame(nav_frame)
        nav_buttons_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(nav_buttons_frame, text="◀ Prev", command=self.prev_svg).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_buttons_frame, text="Next ▶", command=self.next_svg).pack(side=tk.RIGHT, padx=(5, 0))
        
        # current file label
        self.file_label = ttk.Label(nav_frame, text="No file loaded")
        self.file_label.pack(fill=tk.X, pady=2)
                
        # rendering controls section
        render_frame = ttk.LabelFrame(control_frame, text="Rendering")
        render_frame.pack(fill=tk.X, pady=(0, 10))
        
        # sample rate controls with 1, 2, 3 buttons
        sample_frame = ttk.Frame(render_frame)
        sample_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(sample_frame, text="Sample Rate:").pack(side=tk.LEFT)
        self.sample_label = ttk.Label(sample_frame, text="1x")
        self.sample_label.pack(side=tk.RIGHT)
        
        sample_btn_frame = ttk.Frame(render_frame)
        sample_btn_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(sample_btn_frame, text="1", width=3, command=lambda: self.set_sample_rate(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(sample_btn_frame, text="2", width=3, command=lambda: self.set_sample_rate(2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(sample_btn_frame, text="3", width=3, command=lambda: self.set_sample_rate(3)).pack(side=tk.LEFT, padx=2)
        
        # reference toggle
        self.reference_var = tk.BooleanVar()
        self.reference_check = ttk.Checkbutton(render_frame, text="Show Reference", 
                                              variable=self.reference_var,
                                              command=self.toggle_reference)
        self.reference_check.pack(fill=tk.X, pady=2)
        
        # viewport controls section
        viewport_frame = ttk.LabelFrame(control_frame, text="Viewport")
        viewport_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(viewport_frame, text="Reset View", command=self.reset_viewport).pack(fill=tk.X, pady=2)
        
        # viewport info
        self.viewport_label = ttk.Label(viewport_frame, text="Viewport: (0, 0) span=1")
        self.viewport_label.pack(fill=tk.X, pady=2)
        
        # mouse tracking
        self.last_mouse_x = 0
        self.last_mouse_y = 0
    
    def load_file(self):
        """Load a single SVG file."""
        filename = filedialog.askopenfilename(
            title="Select SVG File",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if filename:
            self.svg_files = []
            self.load_svg_file(filename)
            if self.svg_files:
                self.current_svg_index = 0
                self.current_svg = self.svg_files[0][1]
                self.reset_viewport()
                self.render_and_display()
    
    def load_directory(self):
        """Load all SVG files from a directory."""
        directory = filedialog.askdirectory(title="Select SVG Directory")
        if directory:
            self.svg_files = []
            self.load_svg_directory(directory)
            if self.svg_files:
                self.current_svg_index = 0
                self.current_svg = self.svg_files[0][1]
                self.reset_viewport()
                self.render_and_display()
    
    def load_svg_file(self, filename: str) -> bool:
        """Load a single SVG file."""
        try:
            parser = SVGParser()
            svg = parser.parse_file(filename)
            self.svg_files.append((filename, svg))
            print(f"Loaded: {os.path.basename(filename)}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error loading {filename}: {e}")
            return False
    
    def load_svg_directory(self, directory: str):
        """Load all SVG files from a directory."""
        if not os.path.isdir(directory):
            messagebox.showerror("Error", f"Directory not found: {directory}")
            return
        
        svg_files = []
        for filename in os.listdir(directory):
            if filename.lower().endswith('.svg'):
                filepath = os.path.join(directory, filename)
                svg_files.append(filepath)
        
        # Sort files
        svg_files.sort()
        for filepath in svg_files[:9]:
            self.load_svg_file(filepath)
    
    def prev_svg(self):
        """Switch to previous SVG."""
        if len(self.svg_files) > 1:
            self.current_svg_index = (self.current_svg_index - 1) % len(self.svg_files)
            self.current_svg = self.svg_files[self.current_svg_index][1]
            self.reset_viewport()
            self.render_and_display()
    
    def next_svg(self):
        """Switch to next SVG."""
        if len(self.svg_files) > 1:
            self.current_svg_index = (self.current_svg_index + 1) % len(self.svg_files)
            self.current_svg = self.svg_files[self.current_svg_index][1]
            self.reset_viewport()
            self.render_and_display()
    
    def set_sample_rate(self, rate: int):
        """Set sample rate to specific value."""
        if 1 <= rate <= 8:
            self.sample_rate = rate
            self.renderer.set_sample_rate(self.sample_rate)
            self.sample_label.config(text=f"{self.sample_rate}x")
            self.render_and_display()
    
    def toggle_reference(self):
        """Toggle reference mode."""
        self.show_reference = self.reference_var.get()
        print(f"Reference mode: {'ON' if self.show_reference else 'OFF'}")
        self.render_and_display()
    
    def reset_viewport(self):
        """Reset viewport to show entire SVG."""
        if self.current_svg:
            center_x = self.current_svg.width / 2
            center_y = self.current_svg.height / 2
            span = max(self.current_svg.width, self.current_svg.height) / 2
            self.renderer.viewport.set_viewbox(center_x, center_y, span)
            self.update_viewport_label()
    
    def update_viewport_label(self):
        """Update viewport info label."""
        vp = self.renderer.viewport
        self.viewport_label.config(text=f"Viewport: ({vp.x:.1f}, {vp.y:.1f}) span={vp.span:.1f}")
    
    def on_mouse_down(self, event):
        """Handle mouse press."""
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
    
    def on_mouse_drag(self, event):
        """Handle mouse drag for panning."""
        if self.current_svg:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            
            # Convert screen delta to world delta
            scale = self.renderer.viewport.span * 2 / min(self.width, self.height)
            world_dx = -dx * scale
            world_dy = -dy * scale
            
            # Update viewport
            self.renderer.viewport.update_viewbox(world_dx, world_dy, 1.0)
            self.update_viewport_label()
            
            # Re-render
            self.render_and_display()
            
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming."""
        if self.current_svg:
            zoom_factor = 0.9 if event.delta > 0 else 1.1
            
            # Update viewport span
            self.renderer.viewport.update_viewbox(0, 0, zoom_factor)
            self.update_viewport_label()
            
            # Re-render
            self.render_and_display()
    
    def render_and_display(self):
        """Render current SVG and display on canvas."""
        if not self.current_svg:
            return
        
        try:
            # Render SVG
            self.renderer.draw_svg(self.current_svg)
            
            # Reshape buffer to height x width x 4 (RGBA)
            img_array = self.pixel_buffer.reshape((self.height, self.width, 4))
            
            ppm_header = f"P6\n{self.width} {self.height}\n255\n"
            
            # Convert RGBA to RGB bytes
            rgb_data = bytearray()
            for y in range(self.height):
                for x in range(self.width):
                    r, g, b, a = img_array[y, x]
                    rgb_data.extend([r, g, b])
            
            # Create PhotoImage from PPM data
            ppm_data = ppm_header.encode() + rgb_data
            self.photo = tk.PhotoImage(data=ppm_data, format="PPM")
            
            # Display on canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # Update file label
            if self.svg_files:
                filename = os.path.basename(self.svg_files[self.current_svg_index][0])
                self.file_label.config(text=f"File: {filename} ({self.current_svg_index + 1}/{len(self.svg_files)})")
            
        except Exception as e:
            print(f"Error rendering SVG: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Rendering Error", f"Error rendering SVG: {e}")
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SVG Rasterizer - Tkinter Implementation")
    parser.add_argument("path", nargs="?", help="SVG file or directory to load")
    parser.add_argument("--width", type=int, default=800, help="Window width")
    parser.add_argument("--height", type=int, default=600, help="Window height")
    
    args = parser.parse_args()
    
    viewer = SVGViewerTkinter(args.width, args.height)
    
    # Load initial file/directory (if provided)
    if args.path:
        if os.path.isfile(args.path):
            viewer.load_svg_file(args.path)
            if viewer.svg_files:
                viewer.current_svg = viewer.svg_files[0][1]
                viewer.reset_viewport()
                viewer.render_and_display()
        elif os.path.isdir(args.path):
            viewer.load_svg_directory(args.path)
            if viewer.svg_files:
                viewer.current_svg = viewer.svg_files[0][1]
                viewer.reset_viewport()
                viewer.render_and_display()
    
    viewer.run()


if __name__ == "__main__":
    main()