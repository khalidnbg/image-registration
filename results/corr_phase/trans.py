import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

# ========= FONCTIONS UTILITAIRES ========= #
def apply_translation(img, tx, ty):
    """Applique une translation (tx, ty) à une image."""
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

def phase_correlation(img_ref, img_mov):
    """Estimation de translation par corrélation de phase."""
    # Conversion grayscale
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)

    # Normalisation
    img1 = (img_ref - np.mean(img_ref)) / (np.std(img_ref) + 1e-10)
    img2 = (img_mov - np.mean(img_mov)) / (np.std(img_mov) + 1e-10)

    # Padding pour éviter aliasing
    h, w = img1.shape
    new_h, new_w = 2*h - 1, 2*w - 1
    pad1 = np.zeros((new_h, new_w))
    pad2 = np.zeros((new_h, new_w))
    pad1[:h, :w] = img1
    pad2[:h, :w] = img2

    # TF
    F1 = fft2(pad1)
    F2 = fft2(pad2)

    # Spectre croisé normalisé
    R = F1 * np.conj(F2)
    R /= np.abs(R) + 1e-10

    # Corrélation inverse
    phase_corr = np.real(ifft2(R))
    phase_corr = fftshift(phase_corr)

    # Pic max
    max_idx = np.unravel_index(np.argmax(phase_corr), phase_corr.shape)
    max_val = phase_corr[max_idx]

    # Décalage relatif
    center_y, center_x = np.array(phase_corr.shape) // 2
    ty = max_idx[0] - center_y
    tx = max_idx[1] - center_x

    return (tx, ty), max_val, phase_corr

# ========= TEST MULTI-TRANSLATIONS ========= #
if __name__ == "__main__":
    # Charger l’image originale
    img_ref = cv2.imread("results/brain.jpg")

    # Définir 5 translations arbitraires (Tx, Ty)
    translations = [(10, 5), (-15, 8), (20, -10), (-25, -15), (30, 12)]

    results = []

    for (tx_true, ty_true) in translations:
        # Appliquer la translation connue
        img_mov = apply_translation(img_ref, tx_true, ty_true)

        # Estimer la translation par corrélation de phase
        (tx_est, ty_est), peak_val, _ = phase_correlation(img_ref, img_mov)

        # Sauvegarder résultats
        results.append(((tx_true, ty_true), (tx_est, ty_est), peak_val))

    # ======== AFFICHAGE DES RÉSULTATS ======== #
    print("\n📊 Résultats des 5 translations (corrélation de phase):")
    for i, (true_t, est_t, peak) in enumerate(results, 1):
        print(f"Test {i}:")
        print(f"   Translation réelle : {true_t}")
        print(f"   Translation estimée: {est_t}")
        print(f"   Valeur du pic corrélation: {peak:.2f}")
        print("   ----")

    # Optionnel: visualiser un cas
    tx_true, ty_true = translations[0]
    img_mov = apply_translation(img_ref, tx_true, ty_true)
    (tx_est, ty_est), _, phase_corr = phase_correlation(img_ref, img_mov)

    plt.figure(figsize=(15,5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)), plt.title("Référence")
    plt.subplot(132), plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB)), plt.title(f"Mobile (Tx={tx_true}, Ty={ty_true})")
    plt.subplot(133), plt.imshow(phase_corr, cmap="hot"), plt.title(f"Corrélation de Phase\nEstimée: (Tx={tx_est}, Ty={ty_est})")
    plt.show()
