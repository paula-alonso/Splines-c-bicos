import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_contour_and_spline(original_image, points, spline):
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.scatter(points[:, 0], points[:, 1], color='red', s=5, label='Contorno Superior')
    
    x_new = np.linspace(points[:, 0].min(), points[:, 0].max(), 500)
    y_new = spline(x_new)
    plt.plot(x_new, y_new, color='blue', label='Spline Cúbico')

    plt.legend()
    plt.title('Contorno Superior y Spline Cúbico')
    plt.show()
