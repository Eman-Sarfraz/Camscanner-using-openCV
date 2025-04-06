import cv2
import numpy as np
import utlis
import os

# Create output folder if it doesn't exist
if not os.path.exists("Scanned"):
    os.makedirs("Scanned")

# Set to True for live webcam mode; False to load an image from file
webCamFeed = False 
pathImage = "img.jpg"  # Used if webCamFeed is False

# Initialize webcam (only used in live mode)
cap = cv2.VideoCapture(0)
cap.set(10, 160)  # Adjust brightness if needed
heightImg = 640
widthImg = 480

count = 0

while True:
    captured = False
    # --- LIVE MODE ---
    if webCamFeed:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break
        # Display live feed with instructions
        liveDisplay = frame.copy()
        cv2.putText(liveDisplay, "Press 'c' to capture, 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Live Feed", liveDisplay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            img = frame.copy()  # Capture the current frame
            captured = True
        elif key == ord('q'):
            break
        else:
            continue  # Wait until a capture is triggered
    # --- IMAGE FILE MODE ---
    else:
        img = cv2.imread(pathImage)
        if img is None:
            print("Image not found! Please check the file path.")
            break
        captured = True

    if captured:
        # Resize and prepare image
        img = cv2.resize(img, (widthImg, heightImg))
        imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
        
        # Preprocessing: convert to gray and blur
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        
        # --- AUTO CANNY EDGE DETECTION ---
        imgThreshold = utlis.auto_canny(imgBlur)
        
        # Dilation & Erosion to enhance edges
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
        
        # Find contours from the edge image
        imgContours = img.copy()
        imgBigContour = img.copy()
        contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
        
        # Detect the biggest contour which is assumed to be the document
        biggest, maxArea = utlis.biggestContour(contours)
        scannedImg = None
        if biggest.size != 0:
            biggest = utlis.reorder(biggest)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
            imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
            
            # Perspective transformation to "scan" the document
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            
            # Crop margins
            imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0]-20, 20:imgWarpColored.shape[1]-20]
            imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
            
            # Adaptive thresholding for a "scanned" look
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgAdaptiveThre = cv2.adaptiveThreshold(
                imgWarpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
            imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
            imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)
            
            scannedImg = imgWarpColored  # Final scanned image
            
            # Stack images to show each processing step
            imageArray = ([img, imgGray, imgThreshold, imgContours],
                          [imgBigContour, scannedImg, imgWarpGray, imgAdaptiveThre])
        else:
            print("⚠️ No document detected. Please adjust your image conditions.")
            imageArray = ([img, imgGray, imgThreshold, imgContours],
                          [imgBlank, imgBlank, imgBlank, imgBlank])
        
        labels = [["Original", "Gray", "Auto Canny", "Contours"],
                  ["Big Contour", "Warped", "Warp Gray", "Adaptive Thresh"]]
        stackedImage = utlis.stackImages(imageArray, 0.75, labels)
        cv2.imshow("Result", stackedImage)
        
        # Wait for user input to either save or move on
        key2 = cv2.waitKey(0) & 0xFF
        if key2 == ord('s') and scannedImg is not None:
            save_path = f"Scanned/myImage{count}.jpg"
            cv2.imwrite(save_path, scannedImg)
            print("Image saved as:", save_path)
            count += 1
        elif key2 == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
