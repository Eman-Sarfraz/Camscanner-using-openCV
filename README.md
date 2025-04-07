# 🧾 OpenCV CamScanner Clone (Python)

This project is a Python-based clone of popular document scanning apps like **CamScanner**, built using **OpenCV** and **NumPy**. It allows users to scan documents either through a **webcam** or from an **image file**, automatically detects document edges, corrects perspective, and generates a clean scanned version of the document.

---

## 🚀 Features

- 🔍 **Automatic Edge Detection** using adaptive Canny algorithm
- ✏️ **Contour Detection** to identify and isolate the document
- 📐 **Perspective Transformation** to "flatten" the document
- 🧠 **Adaptive Thresholding** for a scanned-paper look
- 🖼️ **Real-time Webcam Mode** or static **Image File Mode**
- 📂 Saves scanned outputs into a local `Scanned/` directory
- 🧱 Side-by-side visualization of each processing step

---

## 📁 Directory Structure

## 🧪 How It Works
Here's a breakdown of the steps used in the scanning process:

Preprocessing:

Converts image to grayscale

Applies Gaussian blur

Uses adaptive Canny edge detection (auto_canny) to find edges

Contour Detection:

Finds all contours in the image

Filters for the largest 4-point contour (assumed to be the document)

Perspective Warp:

Reorders the 4 corner points of the document

Applies a perspective transform to warp the image from an angled view to a top-down view

Post-Processing:

Crops the warped image to remove margins

Converts it to grayscale

Applies adaptive thresholding for a clean "scanned" look

Visualization:

Combines all stages (original, grayscale, edges, warped, thresholded) into one window using stackImages

## note : Create a virtual env to ake it work.
