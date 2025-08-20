import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ========================
# Fonctions MI
# ========================

def calculate_histogram_2d(img1, img2, bins=256):
    """Calcule l'histogramme joint de deux images"""
    img1_int = (img1 * (bins-1)).astype(np.int32)
    img2_int = (img2 * (bins-1)).astype(np.int32)
    img1_flat = img1_int.flatten()
    img2_flat = img2_int.flatten()
    hist_joint, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins,
                                      range=[[0, bins-1], [0, bins-1]])
    return hist_joint

def mutual_information_manual(img1, img2, bins=64):
    """Calcule l'information mutuelle manuellement"""
    hist_joint = calculate_histogram_2d(img1, img2, bins)
    prob_joint = hist_joint / np.sum(hist_joint)
    prob_img1 = np.sum(prob_joint, axis=1)
    prob_img2 = np.sum(prob_joint, axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if prob_joint[i, j] > 0 and prob_img1[i] > 0 and prob_img2[j] > 0:
                mi += prob_joint[i, j] * np.log(prob_joint[i, j] / (prob_img1[i] * prob_img2[j]))
    return mi

def mi_cost_txy(params, img_ref, img_mov):
    """Coût basé sur MI (on minimise -MI)."""
    tx, ty = params
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    img_transformed = cv2.warpAffine(img_mov, M, (img_ref.shape[1], img_ref.shape[0]))
    mi = mutual_information_manual(img_ref, img_transformed, bins=64)
    return -mi

def register_txy_mi(img_ref, img_mov, initial_params=(0.0, 0.0)):
    """Estime Tx, Ty en maximisant MI"""
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0

    result = minimize(mi_cost_txy, initial_params, args=(img_ref, img_mov),
                      method='Powell', options={'maxiter': 300, 'ftol': 1e-6})
    return result.x if result.success else initial_params

def apply_translation_txy(img, tx, ty):
    """Applique une translation Tx, Ty"""
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def calculate_mi_score(img_ref, img_mov):
    """Calcule MI entre deux images"""
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    return mutual_information_manual(img_ref, img_mov, bins=64)

# ========================
# Script principal
# ========================
if __name__ == "__main__":
    img_ref = cv2.imread("results/brain.jpg")

    # 5 translations connues
    translations = [(10, 5), (-15, 8), (20, -10), (-25, -15), (30, 12)]

    for i, (tx_true, ty_true) in enumerate(translations, start=1):
        # Génération de l’image mobile
        img_mov = apply_translation_txy(img_ref, tx_true, ty_true)

        # Score MI avant
        mi_before = calculate_mi_score(img_ref, img_mov)

        # Estimation Tx, Ty
        tx_est, ty_est = register_txy_mi(img_ref, img_mov)

        # Correction inverse
        img_registered = apply_translation_txy(img_mov, -tx_est, -ty_est)

        # Score MI après
        mi_after = calculate_mi_score(img_ref, img_registered)

        # Résultats
        print(f"\nTest {i}:")
        print(f"  Translation appliquée : Tx={tx_true}, Ty={ty_true}")
        print(f"  Translation estimée   : Tx={tx_est:.2f}, Ty={ty_est:.2f}")
        print(f"  MI avant  = {mi_before:.4f}, après = {mi_after:.4f}, gain = {mi_after-mi_before:.4f}")

        # Affichage
        plt.figure(figsize=(12, 4))
        plt.subplot(131), plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)), plt.title("Référence")
        plt.subplot(132), plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB)), plt.title(f"Mobile (Tx={tx_true}, Ty={ty_true})")
        plt.subplot(133), plt.imshow(cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB)), plt.title(f"Recalée (Tx={tx_est:.2f}, Ty={ty_est:.2f})")
        plt.tight_layout()
        plt.show()
