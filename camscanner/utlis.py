import cv2
import numpy as np

# --- Auto Canny Function ---
def auto_canny(image, sigma=0.33):
    # Compute the median of the pixel intensities
    v = np.median(image)
    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

# --- Stack Images ---
def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    
    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray)
    
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(rows):
            for c in range(cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d), 
                              (c * eachImgWidth + len(labels[d]) * 13 + 27, 30 + eachImgHeight * d), 
                              (255, 255, 255), cv2.FILLED)
                label_text = str(labels[d][c])
                cv2.putText(ver, label_text, (c * eachImgWidth + 10, eachImgHeight * d + 20), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

# --- Find the Biggest Contour ---
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Lower minimum area threshold if needed
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

# --- Reorder Points ---
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# --- Draw Rectangle ---
def drawRectangle(img, biggest, thickness):
    cv2.line(img, tuple(biggest[0][0]), tuple(biggest[1][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[0][0]), tuple(biggest[2][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[3][0]), tuple(biggest[2][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[3][0]), tuple(biggest[1][0]), (0, 255, 0), thickness)
    return img
