import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def ssd_cost_txy(params, img_ref, img_mov):
    """Coût SSD en considérant Tx et Ty."""
    tx, ty = params

    # Matrice de translation (Tx, Ty)
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])

    # Appliquer translation
    img_transformed = cv2.warpAffine(
        img_mov, M, (img_ref.shape[1], img_ref.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Calcul du SSD
    diff = img_ref - img_transformed
    return np.sum(diff**2)

def register_txy(img_ref, img_mov, initial_params=(0.0, 0.0)):
    """Estimation de Tx et Ty par SSD."""
    # Conversion en niveaux de gris
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    # Normalisation
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0

    # Optimisation
    result = minimize(
        ssd_cost_txy, initial_params,
        args=(img_ref, img_mov),
        method='Powell',
        options={'maxiter': 300, 'ftol': 1e-6}
    )
    
    return result.x if result.success else initial_params

def apply_translation_txy(img, tx, ty):
    """Applique une translation (Tx, Ty)."""
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# Exemple d’utilisation
if __name__ == "__main__":
    # Charger image de référence
    img_ref = cv2.imread('results/brain.jpg')

    # Définir une série de translations (Tx, Ty)
    translations = [(10, 5), (-15, 8), (20, -10), (-25, -15), (30, 12)]

    # Boucle sur les translations
    for i, (tx_true, ty_true) in enumerate(translations, start=1):
        # Appliquer translation
        img_mov = apply_translation_txy(img_ref, tx_true, ty_true)

        # Estimation
        tx_est, ty_est = register_txy(img_ref, img_mov)

        # Affichage résultats
        print(f"\nTest {i}:")
        print(f"  Translation appliquée : Tx={tx_true}, Ty={ty_true}")
        print(f"  Translation estimée   : Tx={tx_est:.2f}, Ty={ty_est:.2f}")

        # Affichage visuel
        plt.figure(figsize=(12, 4))
        plt.subplot(131), plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)), plt.title("Référence")
        plt.subplot(132), plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB)), plt.title(f"Mobile (Tx={tx_true}, Ty={ty_true})")
        plt.subplot(133), plt.imshow(cv2.cvtColor(apply_translation_txy(img_mov, -tx_est, -ty_est), cv2.COLOR_BGR2RGB)), plt.title(f"Recalée (Tx={tx_est:.2f}, Ty={ty_est:.2f})")
        plt.tight_layout()
        plt.show()
