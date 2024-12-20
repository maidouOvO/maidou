import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
import os
from typing import Dict, List, Tuple
import fitz

class PDFEditor(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PDF Text Editor")
        self.geometry("1000x800")

        # Main container
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Navigation frame
        self.nav_frame = ttk.Frame(self.main_container)
        self.nav_frame.pack(fill=tk.X, pady=(0, 10))

        self.prev_btn = ttk.Button(self.nav_frame, text="Previous", command=self.prev_page)
        self.prev_btn.pack(side=tk.LEFT)

        self.page_label = ttk.Label(self.nav_frame, text="Page: 1")
        self.page_label.pack(side=tk.LEFT, padx=10)

        self.next_btn = ttk.Button(self.nav_frame, text="Next", command=self.next_page)
        self.next_btn.pack(side=tk.LEFT)

        # Canvas for image display
        self.canvas = tk.Canvas(self.main_container, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self.main_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Initialize variables
        self.current_page = 1
        self.images: Dict[int, ImageTk.PhotoImage] = {}
        self.text_boxes: Dict[int, List[Dict]] = {}
        self.current_image = None
        self.output_dir = ""
        self.total_pages = 0

        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def load_pdf(self, output_dir: str):
        """Load processed PDF images and text data"""
        self.output_dir = output_dir

        # Load text data
        try:
            with open(os.path.join(output_dir, "text_coordinates.json"), "r") as f:
                self.text_boxes = json.load(f)
        except FileNotFoundError:
            self.text_boxes = {}

        # Count total pages
        self.total_pages = len([f for f in os.listdir(output_dir)
                              if f.startswith("page_") and f.endswith(".png")])

        self.load_current_page()

    def load_current_page(self):
        """Load the current page image and text boxes"""
        page_file = f"page_{self.current_page:03d}.png"
        image_path = os.path.join(self.output_dir, page_file)

        if os.path.exists(image_path):
            # Load and display image
            image = Image.open(image_path)
            # Scale image to fit canvas width while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width()
            scale = canvas_width / image.width
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

            self.current_image = ImageTk.PhotoImage(image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)

            # Draw text boxes
            page_key = str(self.current_page)
            if page_key in self.text_boxes:
                for idx, text_box in enumerate(self.text_boxes[page_key]):
                    bbox = text_box["bbox"]
                    # Scale coordinates
                    x0, y0, x1, y1 = [coord * scale for coord in bbox]
                    self.canvas.create_rectangle(x0, y0, x1, y1,
                                              outline="blue",
                                              tags=f"text_box_{idx}")

            self.page_label.config(text=f"Page: {self.current_page}/{self.total_pages}")

    def next_page(self):
        """Go to next page"""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.load_current_page()

    def prev_page(self):
        """Go to previous page"""
        if self.current_page > 1:
            self.current_page -= 1
            self.load_current_page()

    def on_canvas_click(self, event):
        """Handle canvas click events"""
        # Get canvas coordinates
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Check if click is within any text box
        page_key = str(self.current_page)
        if page_key in self.text_boxes:
            canvas_width = self.canvas.winfo_width()
            image = Image.open(os.path.join(self.output_dir, f"page_{self.current_page:03d}.png"))
            scale = canvas_width / image.width

            for idx, text_box in enumerate(self.text_boxes[page_key]):
                bbox = text_box["bbox"]
                # Scale coordinates
                x0, y0, x1, y1 = [coord * scale for coord in bbox]

                if x0 <= x <= x1 and y0 <= y <= y1:
                    if messagebox.askyesno("Delete Text Box",
                                         f"Delete text box containing:\n{text_box['text']}"):
                        # Remove text box
                        self.text_boxes[page_key].pop(idx)
                        # Save changes
                        self.save_changes()
                        # Reload page
                        self.load_current_page()
                        break

    def save_changes(self):
        """Save text box changes to JSON file"""
        json_path = os.path.join(self.output_dir, "text_coordinates.json")
        with open(json_path, "w") as f:
            json.dump(self.text_boxes, f, indent=2)

def main(output_dir: str):
    """Start the PDF editor application"""
    app = PDFEditor()
    app.load_pdf(output_dir)
    app.mainloop()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide the output directory path")
