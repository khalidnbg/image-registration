import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def calculate_histogram_2d(img1, img2, bins=64):
    """Calcule l'histogramme joint de deux images"""
    img1_int = (img1 * (bins-1)).astype(np.int32)
    img2_int = (img2 * (bins-1)).astype(np.int32)
    
    img1_flat = img1_int.flatten()
    img2_flat = img2_int.flatten()
    
    hist_joint, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins, range=[[0, bins-1], [0, bins-1]])
    
    return hist_joint

def mutual_information_manual(img1, img2, bins=32):
    """Calcule l'information mutuelle selon la formule MI"""
    hist_joint = calculate_histogram_2d(img1, img2, bins)
    
    # Lissage pour réduire le bruit dans l'histogramme
    hist_joint = hist_joint + 1e-7  # Régularisation
    
    # Conversion en probabilités
    prob_joint = hist_joint / np.sum(hist_joint)
    
    # Probabilités marginales
    prob_img1 = np.sum(prob_joint, axis=1)
    prob_img2 = np.sum(prob_joint, axis=0)
    
    # Calcul MI selon la formule (vectorisé pour éviter les boucles)
    # Éviter log(0) avec un masque
    mask = (prob_joint > 1e-12) & (prob_img1[:, None] > 1e-12) & (prob_img2[None, :] > 1e-12)
    
    mi = 0.0
    if np.any(mask):
        log_term = np.log(prob_joint / (prob_img1[:, None] * prob_img2[None, :]))
        mi = np.sum(prob_joint * log_term * mask)
    
    return mi

def mi_cost(params, img_ref, img_mov, mask=None):
    """Fonction de coût MI (minimise -MI pour maximiser MI)"""
    tx, ty, angle = params
    
    # Matrice de transformation
    M = cv2.getRotationMatrix2D((img_mov.shape[1]//2, img_mov.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Transformation de l'image mobile avec interpolation bilinéaire
    img_transformed = cv2.warpAffine(img_mov, M, (img_ref.shape[1], img_ref.shape[0]), 
                                   flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_REFLECT)
    
    # Application du masque si fourni
    if mask is not None:
        img1_masked = img_ref * mask
        img2_masked = img_transformed * mask
        mask_bool = mask > 0
        if np.sum(mask_bool) > 100:  # Assez de pixels
            img1_masked = img1_masked[mask_bool]
            img2_masked = img2_masked[mask_bool]
        else:
            img1_masked = img_ref
            img2_masked = img_transformed
    else:
        img1_masked = img_ref
        img2_masked = img_transformed
    
    # Calcul MI avec moins de bins pour plus de robustesse au bruit
    mi = mutual_information_manual(img1_masked, img2_masked, bins=32)
    
    return -mi

def register_images_mi(img_ref, img_mov, initial_params=[0, 0, 0]):
    """Recalage par optimisation MI avec multiples tentatives"""
    
    # Conversion en niveaux de gris si nécessaire
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    # Normalisation
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    
    # Multiples points de départ pour éviter minima locaux
    best_params = initial_params
    best_mi = -np.inf
    
    # Points de départ variés
    starts = [
        [0, 0, 0],
        [5, 5, 10], [-5, -5, -10],
        [10, 0, 15], [0, 10, -15],
        [-10, 5, 20], [5, -10, -20]
    ]
    
    for start in starts:
        try:
            result = minimize(mi_cost, start,
                            args=(img_ref, img_mov),
                            method='Powell',
                            options={'maxiter': 1000, 'ftol': 1e-6})
            
            if result.success and -result.fun > best_mi:
                best_mi = -result.fun
                best_params = result.x
        except:
            continue
    
    return best_params

def apply_transformation(img, params):
    """Applique la transformation trouvée"""
    tx, ty, angle = params
    
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def calculate_mi_score(img_ref, img_mov):
    """Calcule le score MI entre deux images"""
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    
    return mutual_information_manual(img_ref, img_mov, bins=32)

# Script principal
if __name__ == "__main__":
    # Charger vos images
    img_ref = cv2.imread('images/tisdrin.png')  # image fixe
    img_mov = cv2.imread('images/tisdrin_translated_12_27.jpg')     # image à recaler
    
    # Score MI avant recalage
    mi_before = calculate_mi_score(img_ref, img_mov)
    print(f"Score MI avant recalage: {mi_before:.4f}")
    
    # Recalage
    params = register_images_mi(img_ref, img_mov)
    print(f"Paramètres trouvés - tx: {params[0]:.2f}, ty: {params[1]:.2f}, angle: {params[2]:.2f}°")
    
    # Application de la transformation
    img_registered = apply_transformation(img_mov, params)
    
    # Score MI après recalage
    mi_after = calculate_mi_score(img_ref, img_registered)
    print(f"Score MI après recalage: {mi_after:.4f}")
    print(f"Amélioration: {mi_after - mi_before:.4f}")
    
    # Affichage des résultats
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    plt.title('Image de référence')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB))
    plt.title(f'Image mobile (avant)\nMI: {mi_before:.3f}')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB))
    plt.title(f'Image mobile (après recalage)\nMI: {mi_after:.3f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Histogrammes joints
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    hist_before = calculate_histogram_2d(
        cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0,
        cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    )
    plt.imshow(hist_before, cmap='hot', origin='lower')
    plt.title(f'Histogramme joint AVANT\nMI: {mi_before:.3f}')
    plt.xlabel('Intensité image mobile')
    plt.ylabel('Intensité image référence')
    
    plt.subplot(122)
    hist_after = calculate_histogram_2d(
        cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0,
        cv2.cvtColor(img_registered, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    )
    plt.imshow(hist_after, cmap='hot', origin='lower')
    plt.title(f'Histogramme joint APRÈS\nMI: {mi_after:.3f}')
    plt.xlabel('Intensité image mobile')
    plt.ylabel('Intensité image référence')
    
    plt.tight_layout()
    plt.show()