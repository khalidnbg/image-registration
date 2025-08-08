import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def ncc_cost(params, img_ref, img_mov, mask=None):
    """Coût NCC (à minimiser car on maximise NCC)."""
    tx, ty, angle = params
    
    # Matrice de transformation
    M = cv2.getRotationMatrix2D((img_mov.shape[1]//2, img_mov.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Transformation avec interpolation bilinéaire + réflexion des bords
    img_transformed = cv2.warpAffine(
        img_mov, M, (img_ref.shape[1], img_ref.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Application du masque si fourni
    if mask is not None:
        img_ref_masked = img_ref * mask
        img_mov_masked = img_transformed * mask
        mask_bool = mask > 0
        img_ref_masked = img_ref_masked[mask_bool]
        img_mov_masked = img_mov_masked[mask_bool]
    else:
        img_ref_masked = img_ref.flatten()
        img_mov_masked = img_transformed.flatten()
    
    # Calcul NCC
    mean_ref = np.mean(img_ref_masked)
    mean_mov = np.mean(img_mov_masked)
    
    numerator = np.sum((img_ref_masked - mean_ref) * (img_mov_masked - mean_mov))
    denominator = np.sqrt(
        np.sum((img_ref_masked - mean_ref) ** 2) * 
        np.sum((img_mov_masked - mean_mov) ** 2)
    )
    
    if denominator == 0:
        return 1.0  # Évite la division par zéro
    
    ncc = numerator / denominator
    return -ncc  # On minimise pour maximiser NCC

def register_images_ncc(img_ref, img_mov, initial_params=[0.0, 0.0, 0.0]):
    """Recalage par NCC avec pyramide d'images + multi-start."""
    # Conversion en niveaux de gris
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    # Normalisation [0, 1]
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0

    # Approche multi-échelle
    scales = [0.25, 0.5, 1.0]  # De basse à haute résolution
    params = np.array(initial_params, dtype=np.float64)
    
    for scale in scales:
        if scale != 1.0:
            ref_scaled = cv2.resize(img_ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            mov_scaled = cv2.resize(img_mov, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            ref_scaled = img_ref
            mov_scaled = img_mov
        
        # Ajustement des paramètres pour l'échelle
        params_scaled = params.copy()
        params_scaled[:2] *= scale

        # Multi-start pour éviter les minima locaux
        starts = [
            params_scaled,
            params_scaled + [5.0, 5.0, 10.0],
            params_scaled + [-5.0, -5.0, -10.0],
            params_scaled + [10.0, 0.0, 15.0],
            params_scaled + [0.0, 10.0, -15.0]
        ]
        
        best_params = params_scaled
        best_ncc = -np.inf  # On maximise NCC
        
        for start in starts:
            try:
                result = minimize(
                    ncc_cost, start,
                    args=(ref_scaled, mov_scaled),
                    method='Powell',
                    options={'maxiter': 500, 'ftol': 1e-6}
                )
                current_ncc = -result.fun  # Car on minimise -NCC
                if result.success and current_ncc > best_ncc:
                    best_ncc = current_ncc
                    best_params = result.x
            except:
                continue
        
        # Mise à jour des paramètres pour l'échelle suivante
        params[:2] = best_params[:2] / scale
        params[2] = best_params[2]
    
    return params

def apply_transformation(img, params):
    """Applique la transformation finale."""
    tx, ty, angle = params
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def calculate_ncc_score(img_ref, img_mov):
    """Calcule le score NCC entre deux images."""
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    return -ncc_cost([0.0, 0.0, 0.0], img_ref, img_mov)  # Retourne NCC (pas -NCC)

# Exemple d'utilisation
if __name__ == "__main__":
    img_ref = cv2.imread('images/tisdrin.png')
    img_mov = cv2.imread('images/tisdrin_translated_12_27.jpg')
    
    # Score avant recalage
    ncc_before = calculate_ncc_score(img_ref, img_mov)
    print(f"NCC avant recalage: {ncc_before:.4f}")
    
    # Recalage
    params = register_images_ncc(img_ref, img_mov)
    print(f"Paramètres trouvés - tx: {params[0]:.2f}, ty: {params[1]:.2f}, angle: {params[2]:.2f}°")
    
    # Application de la transformation
    img_registered = apply_transformation(img_mov, params)
    
    # Score après recalage
    ncc_after = calculate_ncc_score(img_ref, img_registered)
    print(f"NCC après recalage: {ncc_after:.4f}")
    print(f"Amélioration: {ncc_after - ncc_before:.4f}")
    
    # Affichage
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    plt.title('Référence')
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB))
    plt.title(f'Mobile (NCC: {ncc_before:.4f})')
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB))
    plt.title(f'Recalée (NCC: {ncc_after:.4f})')
    plt.show()