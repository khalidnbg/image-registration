import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Erreur : Impossible de charger l'image '{image_path}'")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray_img

def compute_fourier(img):
    f_transform = fft2(img)
    f_transform_shifted = fftshift(f_transform)
    magnitude_spectrum = np.log(1 + np.abs(f_transform_shifted))
    phase_spectrum = np.angle(f_transform_shifted)  
    return magnitude_spectrum, phase_spectrum

def display_images(original, gray, magnitude, phase):
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Image Couleur")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Image en niveaux de gris")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(magnitude, cmap='gray')
    plt.title("Spectre de Magnitude")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(phase, cmap='gray')
    plt.title("Spectre de Phase")
    plt.axis("off")

    plt.show()

image_path = "images/reference_image.jpeg"
original_img, gray_img = load_image(image_path)
magnitude, phase = compute_fourier(gray_img)

display_images(original_img, gray_img, magnitude, phase)
