import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar y convertir la imagen a escala de grises
def load_and_convert_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Aplicar el algoritmo de detección de bordes (Canny)
def detect_edges(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    edges = cv2.Canny(blurred, 10, 80) 
    cv2.imwrite('canny.jpg', edges)
    return edges

# Extraer las coordenadas del contorno superior
def extract_upper_contour(edges, image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(cleaned, kernel, iterations=3)
    closed = cv2.erode(dilated, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))

    largest_contour = contours[0]
    points = largest_contour[:, 0, :]
    height, _ = edges.shape
    upper_points = points[points[:, 1] < height * 0.7]

    unique_points = {}
    for point in upper_points:
        x, y = point
        if x not in unique_points or y < unique_points[x]:
            unique_points[x] = y
    unique_x = np.array(sorted(unique_points.keys()))
    unique_y = np.array([unique_points[x] for x in unique_x])
    unique_points = np.column_stack((unique_x, unique_y))

    return unique_points

    
    

   
    