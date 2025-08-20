import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def ncc_cost_txy(params, img_ref, img_mov):
    """Coût NCC basé uniquement sur Tx et Ty (sans rotation)."""
    tx, ty = params
    
    # Matrice de translation
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    
    img_transformed = cv2.warpAffine(
        img_mov, M, (img_ref.shape[1], img_ref.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Aplatir
    ref = img_ref.flatten()
    mov = img_transformed.flatten()
    
    # Moyennes
    mean_ref = np.mean(ref)
    mean_mov = np.mean(mov)
    
    numerator = np.sum((ref - mean_ref) * (mov - mean_mov))
    denominator = np.sqrt(np.sum((ref - mean_ref)**2) * np.sum((mov - mean_mov)**2))
    
    if denominator == 0:
        return 1.0  # évite division par zéro
    
    ncc = numerator / denominator
    return -ncc  # on minimise -NCC

def register_txy_ncc(img_ref, img_mov, initial_params=(0.0, 0.0)):
    """Estime Tx, Ty en maximisant NCC."""
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0

    result = minimize(
        ncc_cost_txy, initial_params,
        args=(img_ref, img_mov),
        method='Powell',
        options={'maxiter': 300, 'ftol': 1e-6}
    )
    
    return result.x if result.success else initial_params

def apply_translation_txy(img, tx, ty):
    """Applique une translation Tx, Ty."""
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def calculate_ncc_score(img_ref, img_mov):
    """Calcule NCC entre deux images."""
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    
    ref = img_ref.flatten()
    mov = img_mov.flatten()
    mean_ref, mean_mov = np.mean(ref), np.mean(mov)
    
    numerator = np.sum((ref - mean_ref) * (mov - mean_mov))
    denominator = np.sqrt(np.sum((ref - mean_ref)**2) * np.sum((mov - mean_mov)**2))
    return numerator / denominator if denominator != 0 else 0

# Exemple d'utilisation
if __name__ == "__main__":
    img_ref = cv2.imread("results/brain.jpg")

    # Définir 5 translations connues
    translations = [(10, 5), (-15, 8), (20, -10), (-25, -15), (30, 12)]

    for i, (tx_true, ty_true) in enumerate(translations, start=1):
        # Appliquer translation
        img_mov = apply_translation_txy(img_ref, tx_true, ty_true)

        # Score avant
        ncc_before = calculate_ncc_score(img_ref, img_mov)

        # Estimation
        tx_est, ty_est = register_txy_ncc(img_ref, img_mov)

        # Appliquer correction
        img_registered = apply_translation_txy(img_mov, -tx_est, -ty_est)

        # Score après
        ncc_after = calculate_ncc_score(img_ref, img_registered)

        # Résultats
        print(f"\nTest {i}:")
        print(f"  Translation appliquée : Tx={tx_true}, Ty={ty_true}")
        print(f"  Translation estimée   : Tx={tx_est:.2f}, Ty={ty_est:.2f}")
        print(f"  NCC avant  = {ncc_before:.4f}, après = {ncc_after:.4f}, gain = {ncc_after-ncc_before:.4f}")

        # Affichage
        plt.figure(figsize=(12, 4))
        plt.subplot(131), plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)), plt.title("Référence")
        plt.subplot(132), plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB)), plt.title(f"Mobile (Tx={tx_true}, Ty={ty_true})")
        plt.subplot(133), plt.imshow(cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB)), plt.title(f"Recalée (Tx={tx_est:.2f}, Ty={ty_est:.2f})")
        plt.tight_layout()
        plt.show()
