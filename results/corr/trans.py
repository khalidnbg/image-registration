import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
import cv2
import pandas as pd

# ==========================
# Fonctions utilitaires
# ==========================

def apply_translation(img, tx, ty):
    """Applique une translation Tx, Ty à une image."""
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def preprocess_images(img1, img2, normalize=True):
    """Prépare deux images (redimension + normalisation)."""
    img1_proc = img1.copy().astype(np.float64)
    img2_proc = img2.copy().astype(np.float64)

    # Redimensionner si tailles différentes
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1_proc = cv2.resize(img1_proc, (min_w, min_h))
        img2_proc = cv2.resize(img2_proc, (min_w, min_h))

    if normalize:
        img1_proc = (img1_proc - np.mean(img1_proc)) / np.std(img1_proc)
        img2_proc = (img2_proc - np.mean(img2_proc)) / np.std(img2_proc)

    return img1_proc, img2_proc

def correlation_fft(img1, img2):
    """Calcule la corrélation croisée par FFT et retourne (tx, ty, confiance)."""
    img1_proc, img2_proc = preprocess_images(img1, img2)

    h, w = img1_proc.shape
    new_h, new_w = h*2, w*2

    # Padding
    img1_pad = np.zeros((new_h, new_w))
    img2_pad = np.zeros((new_h, new_w))
    img1_pad[:h, :w] = img1_proc
    img2_pad[:h, :w] = img2_proc

    # FFT
    F1 = fft2(img1_pad)
    F2 = fft2(img2_pad)
    C = F1 * np.conj(F2)

    # Corrélation
    correlation = np.real(ifft2(C))
    correlation = fftshift(correlation)

    # Pic max
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    max_val = correlation[max_idx]

    # Translation détectée
    center_y, center_x = np.array(correlation.shape) // 2
    ty = max_idx[0] - center_y
    tx = max_idx[1] - center_x

    # Confiance
    confidence = max_val / np.mean(correlation)

    return tx, ty, confidence

# ==========================
# Script principal
# ==========================
if __name__ == "__main__":
    img_ref = cv2.imread("results/brain.jpg", cv2.IMREAD_GRAYSCALE)

    # 5 translations connues
    translations = [(10, 5), (-15, 8), (20, -10), (-25, -15), (30, 12)]

    results = []

    for i, (tx_true, ty_true) in enumerate(translations, start=1):
        # Appliquer translation
        img_mov = apply_translation(img_ref, tx_true, ty_true)

        # Détection par corrélation
        tx_est, ty_est, confidence = correlation_fft(img_ref, img_mov)

        # Sauvegarde
        results.append({
            "Test": i,
            "Tx appliqué": tx_true,
            "Ty appliqué": ty_true,
            "Tx estimé": tx_est,
            "Ty estimé": ty_est,
            "Confiance": confidence
        })

        # Affichage rapide
        print(f"\nTest {i}:")
        print(f"  Appliqué : Tx={tx_true}, Ty={ty_true}")
        print(f"  Estimé   : Tx={tx_est}, Ty={ty_est}")
        print(f"  Confiance = {confidence:.2f}")

        # Visualisation
        plt.figure(figsize=(12,4))
        plt.subplot(131), plt.imshow(img_ref, cmap="gray"), plt.title("Référence")
        plt.subplot(132), plt.imshow(img_mov, cmap="gray"), plt.title(f"Mobile (Tx={tx_true}, Ty={ty_true})")
        img_reg = apply_translation(img_mov, -tx_est, -ty_est)
        plt.subplot(133), plt.imshow(img_reg, cmap="gray"), plt.title(f"Recalée (Tx={tx_est}, Ty={ty_est})")
        plt.tight_layout()
        plt.show()

    # Tableau récapitulatif
    df = pd.DataFrame(results)
    print("\n=== Résumé Corrélation FFT ===")
    print(df.to_string(index=False))
