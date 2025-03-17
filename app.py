import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu, threshold_multiotsu
from sklearn.cluster import KMeans
import os

# Streamlit app title
st.title("Facial Edge Detection and Segmentation")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the original image
    st.subheader("Original Image")
    st.image(image_rgb, use_column_width=True)

    # Preprocessing
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gaussian_blurred = cv2.GaussianBlur(gray, (21, 21), 4)
    median_blurred_2 = cv2.medianBlur(gray, 25)

    # Edge Detection Functions
    def sobel_edge_detection(img):
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        sobel_x = cv2.filter2D(img, -1, sobel_x_kernel)
        sobel_y = cv2.filter2D(img, -1, sobel_y_kernel)
        sobel_magnitude = np.hypot(sobel_x, sobel_y)
        return np.uint8(sobel_magnitude / np.max(sobel_magnitude) * 255)

    def scharr_edge_detection(img):
        scharr_x_kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
        scharr_y_kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)
        scharr_x = cv2.filter2D(img, -1, scharr_x_kernel)
        scharr_y = cv2.filter2D(img, -1, scharr_y_kernel)
        scharr_magnitude = np.hypot(scharr_x, scharr_y)
        return np.uint8(scharr_magnitude / np.max(scharr_magnitude) * 255)

    def laplacian_edge_detection(img):
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        laplacian = cv2.filter2D(img, -1, laplacian_kernel)
        return np.uint8(np.absolute(laplacian) / np.max(np.absolute(laplacian)) * 255)

    def canny_edge_detection(img):
        grad_x = cv2.filter2D(img, -1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32))
        grad_y = cv2.filter2D(img, -1, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32))
        grad_magnitude = np.hypot(grad_x, grad_y)
        grad_magnitude = grad_magnitude / np.max(grad_magnitude) * 255
        grad_angle = np.arctan2(grad_y, grad_x) * 180 / np.pi % 180

        edges = grad_magnitude.copy()
        rows, cols = edges.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                angle = grad_angle[i, j]
                if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                    neighbors = [edges[i, j-1], edges[i, j+1]]
                elif 22.5 <= angle < 67.5:
                    neighbors = [edges[i-1, j+1], edges[i+1, j-1]]
                elif 67.5 <= angle < 112.5:
                    neighbors = [edges[i-1, j], edges[i+1, j]]
                else:
                    neighbors = [edges[i-1, j-1], edges[i+1, j+1]]
                if edges[i, j] < max(neighbors):
                    edges[i, j] = 0

        low_threshold, high_threshold = 20, 60
        canny_edges = np.zeros_like(edges, dtype=np.uint8)
        strong_edges = edges > high_threshold
        weak_edges = (edges >= low_threshold) & (edges <= high_threshold)
        canny_edges[strong_edges] = 255
        canny_edges[weak_edges] = 128
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if canny_edges[i, j] == 128:
                    if np.any(canny_edges[i-1:i+2, j-1:j+2] == 255):
                        canny_edges[i, j] = 255
                    else:
                        canny_edges[i, j] = 0
        return canny_edges

    def roberts_edge_detection(img):
        roberts_kernel_45 = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_kernel_135 = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        grad_45 = cv2.filter2D(img, -1, roberts_kernel_45)
        grad_135 = cv2.filter2D(img, -1, roberts_kernel_135)
        roberts_edges = np.sqrt(grad_45**2 + grad_135**2)
        return np.clip(roberts_edges, 0, 255).astype(np.uint8)

    def prewitt_edge_detection(img):
        prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x_prewitt = cv2.filter2D(img, -1, prewitt_kernel_x)
        grad_y_prewitt = cv2.filter2D(img, -1, prewitt_kernel_y)
        prewitt_edges = np.sqrt(grad_x_prewitt**2 + grad_y_prewitt**2)
        return np.clip(prewitt_edges, 0, 255).astype(np.uint8)

    # Perform Edge Detection
    st.subheader("Edge Detection Results")
    sobel_magnitude = sobel_edge_detection(gaussian_blurred)
    scharr_magnitude = scharr_edge_detection(gaussian_blurred)
    laplacian = laplacian_edge_detection(gaussian_blurred)
    canny_edges = canny_edge_detection(gaussian_blurred)
    roberts_edges = roberts_edge_detection(gaussian_blurred)
    prewitt_edges = prewitt_edge_detection(gaussian_blurred)

    # Display Edge Detection Results
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    axes[0].imshow(sobel_magnitude, cmap='gray')
    axes[0].set_title("Sobel Edges")
    axes[0].axis("off")
    axes[1].imshow(scharr_magnitude, cmap='gray')
    axes[1].set_title("Scharr Edges")
    axes[1].axis("off")
    axes[2].imshow(laplacian, cmap='gray')
    axes[2].set_title("Laplacian Edges")
    axes[2].axis("off")
    axes[3].imshow(canny_edges, cmap='gray')
    axes[3].set_title("Canny Edges")
    axes[3].axis("off")
    axes[4].imshow(roberts_edges, cmap='gray')
    axes[4].set_title("Roberts Edges")
    axes[4].axis("off")
    axes[5].imshow(prewitt_edges, cmap='gray')
    axes[5].set_title("Prewitt Edges")
    axes[5].axis("off")
    st.pyplot(fig)

    # Segmentation Functions
    st.subheader("Segmentation Results")

    # Thresholding
    manual_threshold = 80
    manual_thresh = (gaussian_blurred > manual_threshold).astype(np.uint8) * 255
    _, otsu_thresh = cv2.threshold(gaussian_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresholds = threshold_multiotsu(gaussian_blurred, classes=3)
    regions = np.digitize(gaussian_blurred, bins=otsu_thresholds)

    # K-Means Clustering
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(pixels)
    segmented = kmeans.cluster_centers_[labels].reshape(image_rgb.shape).astype(np.uint8)

    # HSV Segmentation
    image_hsv = rgb2hsv(image_rgb)
    lower_mask_value, upper_mask_value = 0.001, 0.090
    lower_mask = image_hsv[:, :, 0] > lower_mask_value
    upper_mask = image_hsv[:, :, 0] < upper_mask_value
    mask = lower_mask * upper_mask
    red = image_rgb[:, :, 0] * mask
    green = image_rgb[:, :, 1] * mask
    blue = image_rgb[:, :, 2] * mask
    segmented_image = np.dstack((red, green, blue))

    # MTCNN Face Detection
    detector = MTCNN(thresholds=[0.6, 0.7, 0.9])
    mtcnn_result = image_rgb.copy()
    boxes, probs, landmarks = detector.detect(image_rgb, landmarks=True)
    fig_mtcnn, ax_mtcnn = plt.subplots(figsize=(5, 3))
    ax_mtcnn.imshow(mtcnn_result)
    ax_mtcnn.axis('off')
    if boxes is not None:
        for box, prob, landmark in zip(boxes, probs, landmarks):
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax_mtcnn.add_patch(rect)
            ax_mtcnn.text(x1, y1 - 10, f"{prob:.2f}", color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            ax_mtcnn.scatter(landmark[:, 0], landmark[:, 1], s=8, c='b')

    # Haar Cascade Face Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    haar_result = image_rgb.copy()
    faces_haar = face_cascade.detectMultiScale(image_rgb, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20))
