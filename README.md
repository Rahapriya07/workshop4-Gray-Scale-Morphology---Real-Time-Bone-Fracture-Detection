# WORKSHOP 4 – GRAY SCALE MORPHOLOGY: REAL-TIME BONE FRACTURE DETECTION
## Name : Raha Priya Dharshini M
## Reg No. : 212224240124


## Aim
To implement grayscale morphological operations for image processing and demonstrate a basic bone fracture detection approach using OpenCV and Python.

## Software Requirements
- Python
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

## Procedure

1. Import the required libraries: OpenCV, NumPy, and Matplotlib.
2. Read the input image using `cv2.imread()`.
3. Convert the input image into grayscale using `cv2.cvtColor()`.
4. Apply Gaussian Blur using `cv2.GaussianBlur()` to reduce noise.
5. Create a 5 × 5 morphological kernel using `np.ones()`.
6. Apply **Erosion** using `cv2.erode()` to process the image.
7. Apply **Dilation** using `cv2.dilate()` to refine the morphological result.
8. Perform edge detection using the Canny Edge Detector with threshold values 50 and 150.
9. Find contours using `cv2.findContours()`.
10. Draw the detected contours on the original image using `cv2.drawContours()`.
11. Display the original image and processed image using Matplotlib.

## Code
```

import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_fractures(preprocessed, original):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(preprocessed, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    edges = cv2.Canny(dilation, 50, 150)

    contours, _ = cv2.findContours(
        edges.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = original.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return result

def present_results(original_image, processed_image):
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Fracture Detected Image")
    plt.imshow(processed_rgb)
    plt.axis('off')

    plt.show()

# Main Execution
image_path = 'myfoto.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the file path.")
else:
    preprocessed = preprocess_image(image)
    fracture_detected_image = detect_fractures(preprocessed, image)
    present_results(image, fracture_detected_image)
```

## Output

### Original Image and Fracture Detected Image

<img width="1378" height="641" alt="image" src="https://github.com/user-attachments/assets/9fa1c20c-3ea0-427f-bd15-b6e73c7ed39e" />


## Result
Thus, grayscale morphological operations, edge detection, and contour detection were implemented using OpenCV and Python to demonstrate a basic bone fracture detection approach.
