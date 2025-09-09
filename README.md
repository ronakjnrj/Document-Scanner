# Document Scanner App

This is a Document Scanner application designed to detect and scan documents using your device's camera or image files. It automatically detects edges, crops the document area, applies perspective transformation, and enhances the image to create clean, high-quality scans, similar to a physical scanner.

## ✨ Features

* **📷 Capture or Upload:** Use your device camera or select an image from your gallery. 
* **📐 Automatic Edge Detection:** Detects document boundaries using OpenCV.
* **🧠 Perspective Correction:** Corrects the document's angle for a professional-looking scan.
* **🎨 Image Enhancement:** Improves readability with various filters (grayscale, thresholding, etc.). 
* **📄 Save as PDF or Image:** Exports scanned documents as image files or compiles them into PDFs.   
* **🖱️ Easy-to-use Interface:** Provides a simple GUI for a smooth user experience. 

## 🛠️ Tech Stack  
  
* Python
* OpenCV
* Tkinter / PyQt (depending on GUI implementation) 
* NumPy 
* Pillow (PIL)

## 🚀 Getting Started 

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/document-scanner.git](https://github.com/your-username/document-scanner.git)
    ```

2.  **Install the dependencies:**
 
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**

    ```bash
    python scanner_app.py 
    ```

## 📸 Screenshots 
v 
_(Screenshot will be added soon)_

## 💡 Future Improvements

* OCR (Text Extraction)
* Cloud Save / Share Options
* Multi-page PDF support
* Mobile version with Flutter or React Native

## 🙌 Contribution

Pull requests are welcome! For major changes, please open an issue first to discuss the proposed improvements.
