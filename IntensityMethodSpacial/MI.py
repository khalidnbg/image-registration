# Importation des bibliothèques nécessaires
import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def calculate_histogram_2d(img1, img2, bins=256):
    """Calcule l'histogramme joint de deux images"""
    # Conversion en entiers pour l'histogramme
    img1_int = (img1 * (bins-1)).astype(np.int32)
    img2_int = (img2 * (bins-1)).astype(np.int32)
    
    # Aplatir les images en vecteurs 1D
    img1_flat = img1_int.flatten()
    img2_flat = img2_int.flatten()
    
    # Calcul de l'histogramme joint (2D)
    hist_joint, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins, range=[[0, bins-1], [0, bins-1]])
    
    return hist_joint

def mutual_information_manual(img1, img2, bins=64):
    """Calcule l'information mutuelle manuellement selon la formule"""
    # Calcul de l'histogramme joint
    hist_joint = calculate_histogram_2d(img1, img2, bins)
    
    # Conversion en probabilités (normalisation)
    prob_joint = hist_joint / np.sum(hist_joint)
    
    # Calcul des probabilités marginales
    prob_img1 = np.sum(prob_joint, axis=1)  # somme sur les colonnes
    prob_img2 = np.sum(prob_joint, axis=0)  # somme sur les lignes
    
    # Calcul de l'information mutuelle selon la formule
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if prob_joint[i, j] > 0:  # éviter log(0)
                # Calcul du terme : p(x,y) * log(p(x,y) / (p(x) * p(y)))
                if prob_img1[i] > 0 and prob_img2[j] > 0:
                    mi += prob_joint[i, j] * np.log(prob_joint[i, j] / (prob_img1[i] * prob_img2[j]))
    
    return mi

def mi_cost(params, img_ref, img_mov, mask=None):
    """Fonction qui calcule le coût MI (on veut maximiser MI, donc minimiser -MI)"""
    # Récupération des paramètres de transformation
    tx, ty, angle = params  # translation en x, y et rotation
    
    # Création de la matrice de transformation (rotation autour du centre)
    M = cv2.getRotationMatrix2D((img_mov.shape[1]//2, img_mov.shape[0]//2), angle, 1.0)
    # Ajout de la translation à la matrice
    M[0, 2] += tx  # translation en x
    M[1, 2] += ty  # translation en y
    
    # Application de la transformation à l'image mobile
    img_transformed = cv2.warpAffine(img_mov, M, (img_ref.shape[1], img_ref.shape[0]))
    
    # Application du masque si fourni
    if mask is not None:
        img1_masked = img_ref * mask
        img2_masked = img_transformed * mask
        # Sélectionner seulement les pixels du masque pour le calcul
        mask_bool = mask > 0
        img1_masked = img1_masked[mask_bool]
        img2_masked = img2_masked[mask_bool]
    else:
        img1_masked = img_ref
        img2_masked = img_transformed
    
    # Calcul de l'information mutuelle
    mi = mutual_information_manual(img1_masked, img2_masked, bins=64)
    
    # Retourner -MI car on veut maximiser MI (minimize cherche le minimum)
    return -mi

def register_images_mi(img_ref, img_mov, initial_params=[0, 0, 0]):
    """Fonction principale qui trouve les meilleurs paramètres de recalage par MI"""
    
    # Conversion des images couleur en niveaux de gris
    if len(img_ref.shape) == 3:  # si l'image a 3 canaux (RGB)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    # Normalisation des valeurs entre 0 et 1 (au lieu de 0-255)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    
    # Optimisation : cherche les paramètres qui maximisent MI (minimisent -MI)
    result = minimize(mi_cost, initial_params,  # commence avec [0,0,0]
                     args=(img_ref, img_mov),   # passe les images à la fonction
                     method='Powell',           # algorithme d'optimisation
                     options={'maxiter': 1000}) # maximum 1000 itérations
    
    return result.x  # retourne les meilleurs paramètres trouvés

def apply_transformation(img, params):
    """Applique la transformation trouvée à une image"""
    tx, ty, angle = params  # récupère les paramètres
    
    # Création de la même matrice de transformation
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Application de la transformation
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def calculate_mi_score(img_ref, img_mov):
    """Calcule le score MI final entre deux images (pour vérification)"""
    # Conversion en niveaux de gris si nécessaire
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    # Normalisation
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    
    # Calcul MI
    return mutual_information_manual(img_ref, img_mov, bins=64)

# Script principal
if __name__ == "__main__":
    # 1. Charger vos images (remplacez par vos chemins)
    img_ref = cv2.imread('images/number.png')  # image fixe
    img_mov = cv2.imread('images/tisdrin_translated_12_27.jpg')     # image à recaler
    
    # Score MI avant recalage
    mi_before = calculate_mi_score(img_ref, img_mov)
    print(f"Score MI avant recalage: {mi_before:.4f}")
    
    # 2. Lancer le recalage (trouve les meilleurs paramètres)
    params = register_images_mi(img_ref, img_mov)
    print(f"Paramètres trouvés - tx: {params[0]:.2f}, ty: {params[1]:.2f}, angle: {params[2]:.2f}°")
    
    # 3. Appliquer la transformation à l'image mobile
    img_registered = apply_transformation(img_mov, params)
    
    # Score MI après recalage
    mi_after = calculate_mi_score(img_ref, img_registered)
    print(f"Score MI après recalage: {mi_after:.4f}")
    print(f"Amélioration: {mi_after - mi_before:.4f}")
    
    # 4. Affichage des résultats (avant/après)
    plt.figure(figsize=(15, 5))
    
    # Image de référence
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    plt.title('Image de référence')
    plt.axis('off')
    
    # Image mobile originale
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB))
    plt.title(f'Image mobile (avant)\nMI: {mi_before:.3f}')
    plt.axis('off')
    
    # Image mobile après recalage
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB))
    plt.title(f'Image mobile (après recalage)\nMI: {mi_after:.3f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Optionnel : Affichage de l'histogramme joint pour visualiser
    plt.figure(figsize=(12, 4))
    
    # Histogramme joint avant recalage
    plt.subplot(121)
    hist_before = calculate_histogram_2d(
        cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0,
        cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    )
    plt.imshow(hist_before, cmap='hot', origin='lower')
    plt.title(f'Histogramme joint AVANT\nMI: {mi_before:.3f}')
    plt.xlabel('Intensité image mobile')
    plt.ylabel('Intensité image référence')
    
    # Histogramme joint après recalage
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