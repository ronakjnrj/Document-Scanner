import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.filters import threshold_local
import imutils
import os
import threading

class DocumentScanner:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Scanner")
        self.root.geometry("800x600")
        self.root.configure(padx=10, pady=10)
        
        # Variables
        self.image_path = None
        self.output_path = None
        self.original_image = None
        self.scanned_image = None
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Frame for images
        self.images_frame = tk.Frame(self.root)
        self.images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image frame
        self.original_frame = tk.LabelFrame(self.images_frame, text="Original Image")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.original_label = tk.Label(self.original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        
        # Scanned image frame
        self.scanned_frame = tk.LabelFrame(self.images_frame, text="Scanned Image")
        self.scanned_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.scanned_label = tk.Label(self.scanned_frame)
        self.scanned_label.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        self.select_btn = tk.Button(self.control_frame, text="Select Image", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.scan_btn = tk.Button(self.control_frame, text="Scan Document", command=self.scan_document, state=tk.DISABLED)
        self.scan_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(self.control_frame, text="Save Scanned Image", command=self.save_image, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient=tk.HORIZONTAL, length=200, mode="indeterminate")
        self.progress.pack(side=tk.RIGHT, padx=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Select an image to begin", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if self.image_path:
            self.status_label.config(text=f"Image selected: {os.path.basename(self.image_path)}")
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.original_image, self.original_label)
            self.scan_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)
            self.scanned_image = None
            self.display_image(None, self.scanned_label)
    
    def scan_document(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        self.status_label.config(text="Processing image...")
        self.progress.start()
        self.scan_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.DISABLED)
        
        # Run scanning in a separate thread to keep UI responsive
        threading.Thread(target=self._scan_process, daemon=True).start()
    
    def _scan_process(self):
        try:
            # Clone the original image
            image = self.original_image.copy()
            ratio = image.shape[0] / 500.0
            image = imutils.resize(image, height=500)
            
            # Convert to grayscale and find edges
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(gray, 75, 200)
            
            # Find contours
            cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            
            # Find document contour
            screenCnt = None
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                if len(approx) == 4:
                    screenCnt = approx
                    break
            
            if screenCnt is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not find document contour"))
                self.root.after(0, self._reset_ui)
                return
            
            # Apply perspective transform
            warped = self.four_point_transform(self.original_image, screenCnt.reshape(4, 2) * ratio)
            
            # Convert to grayscale and apply threshold
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            T = threshold_local(warped, 11, offset=10, method="gaussian")
            warped = (warped > T).astype("uint8") * 255
            
            # Store and display result
            self.scanned_image = warped
            self.root.after(0, lambda: self.display_image(self.scanned_image, self.scanned_label))
            self.root.after(0, lambda: self.status_label.config(text="Document scanned successfully"))
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred during scanning: {str(e)}"))
        
        self.root.after(0, self._reset_ui)
    
    def _reset_ui(self):
        self.progress.stop()
        self.scan_btn.config(state=tk.NORMAL)
        self.select_btn.config(state=tk.NORMAL)
    
    def save_image(self):
        if self.scanned_image is None:
            messagebox.showerror("Error", "No scanned image to save")
            return
        
        self.output_path = filedialog.asksaveasfilename(
            title="Save Scanned Image",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")],
            defaultextension=".jpg"
        )
        
        if self.output_path:
            try:
                cv2.imwrite(self.output_path, self.scanned_image)
                messagebox.showinfo("Success", "Image saved successfully ")
                self.status_label.config(text=f"Image saved to: {os.path.basename(self.output_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def display_image(self, cv_image, label_widget):
        if cv_image is None:
            label_widget.config(image='')
            return
        
        # Resize for display
        max_size = 350  # Maximum dimension for display
        h, w = cv_image.shape[:2]
        aspect = w / h
        
        if h > w:
            new_h = max_size
            new_w = int(aspect * new_h)
        else:
            new_w = max_size
            new_h = int(new_w / aspect)
        
        resized = cv2.resize(cv_image, (new_w, new_h))
        
        # Convert from OpenCV BGR to RGB format
        if len(resized.shape) == 3:  # Color image
            image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:  # Grayscale image
            image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            
        # Convert to PhotoImage
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update label
        label_widget.config(image=photo)
        label_widget.image = photo  # Keep reference to prevent garbage collection
    
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentScanner(root)
    root.mainloop()