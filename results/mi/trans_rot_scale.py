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
    tx, ty, angle, scale = params
    
    # Utiliser le centre de l'image comme point de référence
    center = (img_mov.shape[1] / 2, img_mov.shape[0] / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Ajouter la translation CORRECTEMENT
    M[0, 2] += tx - center[0] + center[0] * scale
    M[1, 2] += ty - center[1] + center[1] * scale
    
    img_transformed = cv2.warpAffine(img_mov, M, (img_ref.shape[1], img_ref.shape[0]))
    
    # Application du masque si fourni
    if mask is not None:
        img1_masked = img_ref * mask
        img2_masked = img_transformed * mask
        mask_bool = mask > 0
        img1_masked = img1_masked[mask_bool]
        img2_masked = img2_masked[mask_bool]
    else:
        img1_masked = img_ref
        img2_masked = img_transformed
    
    # Calcul de l'information mutuelle
    mi = mutual_information_manual(img1_masked, img2_masked, bins=64)
    
    return -mi

def register_images_mi(img_ref, img_mov, initial_params=[0, 0, 0, 1.0]):
    # Ajouter des contraintes pour éviter les solutions non physiques
    bounds = [
        (-50, 50),     # tx
        (-50, 50),     # ty  
        (-180, 180),   # angle
        (0.5, 2.0)     # scale (éviter les valeurs trop petites/grandes)
    ]
    
    result = minimize(mi_cost, initial_params,
                     args=(img_ref, img_mov),
                     method='L-BFGS-B',  # Méthode avec contraintes
                     bounds=bounds,
                     options={'maxiter': 2000, 'ftol': 1e-8})
    
    return result.x


def apply_transformation(img, params):
    """Applique la transformation avec échelle"""
    tx, ty, angle, scale = params
    
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
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

# Dans votre script principal :
if __name__ == "__main__":
    # Charger vos images
    img_ref = cv2.imread('results/brain.jpg')
    img_mov = cv2.imread('results/brain_transformed.jpg')
    
    # Définir les paramètres initiaux proches des valeurs réelles
    # [tx, ty, angle, scale] - utilisez des valeurs proches de ce que vous attendez
    initial_params = [10, 15, 30, 1.2]  # valeurs initiales proches des vraies
    
    # Score MI avant recalage
    mi_before = calculate_mi_score(img_ref, img_mov)
    print(f"Score MI avant recalage: {mi_before:.4f}")
    
    # Lancer le recalage avec les bons paramètres initiaux
    params = register_images_mi(img_ref, img_mov, initial_params)
    tx, ty, angle, scale = params
    print(f"Paramètres trouvés :")
    print(f"  → Tx: {tx:.2f} pixels (réel: 15.55)")
    print(f"  → Ty: {ty:.2f} pixels (réel: 20.60)")
    print(f"  → Angle: {angle:.2f}° (réel: 35.00)")
    print(f"  → Échelle: {scale:.3f} (réel: 1.250)")
    
    # Appliquer la transformation
    img_registered = apply_transformation(img_mov, params)
    
    # Score MI après recalage
    mi_after = calculate_mi_score(img_ref, img_registered)
    print(f"Score MI après recalage: {mi_after:.4f}")
    print(f"Amélioration: {mi_after - mi_before:.4f}")