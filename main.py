from image_processing import load_and_convert_image, detect_edges, extract_upper_contour
from spline_interpolation import interpolate_with_cubic_splines
from plotting import plot_contour_and_spline
import cv2

# Función principal
def main(image_path):
    gray_image = load_and_convert_image(image_path)
    edges = detect_edges(gray_image)
    original_image = cv2.imread(image_path)
    points = extract_upper_contour(edges, original_image)
    cs = interpolate_with_cubic_splines(points)
   
    plot_contour_and_spline(original_image, points, cs)

# Ejecutar el programa con una imagen de ejemplo
if __name__ == "__main__":
    image_path = 'assets/moto.jpg'  # Cambia esta ruta por la de tu imagen
    main(image_path)