import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def ssd_cost(params, img_ref, img_mov, mask=None):
    """Coût SSD avec interpolation bilinéaire et gestion des bords."""
    tx, ty, angle = params
    
    # Matrice de transformation
    M = cv2.getRotationMatrix2D((img_mov.shape[1]//2, img_mov.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Transformation avec interpolation lisse et réflexion des bords
    img_transformed = cv2.warpAffine(
        img_mov, M, (img_ref.shape[1], img_ref.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Calcul SSD avec masque optionnel
    if mask is not None:
        diff = (img_ref - img_transformed) * mask
    else:
        diff = img_ref - img_transformed
    
    return np.sum(diff**2)

def register_images_ssd(img_ref, img_mov, initial_params=[0.0, 0.0, 0.0]):
    """Recalage SSD amélioré avec multi-start et pyramide d'images."""
    # Conversion en niveaux de gris
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    # Normalisation [0, 1]
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0

    # Approche multi-échelle (pyramide)
    scales = [0.25, 0.5, 1.0]  # Résolutions réduites -> résolution originale
    params = np.array(initial_params, dtype=np.float64)  # <-- Correction: float64
    
    for scale in scales:
        # Redimensionnement des images
        if scale != 1.0:
            ref_scaled = cv2.resize(img_ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            mov_scaled = cv2.resize(img_mov, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            ref_scaled = img_ref
            mov_scaled = img_mov
        
        # Ajustement des paramètres pour l'échelle (en float)
        params_scaled = params.copy()
        params_scaled[:2] *= scale  # <-- Pas d'erreur de type car float64

        # Multi-start pour éviter les minima locaux
        starts = [
            params_scaled,
            params_scaled + [5.0, 5.0, 10.0],
            params_scaled + [-5.0, -5.0, -10.0],
            params_scaled + [10.0, 0.0, 15.0],
            params_scaled + [0.0, 10.0, -15.0]
        ]
        
        best_params = params_scaled
        best_ssd = np.inf
        
        for start in starts:
            try:
                result = minimize(
                    ssd_cost, start,
                    args=(ref_scaled, mov_scaled),
                    method='Powell',
                    options={'maxiter': 500, 'ftol': 1e-6}
                )
                if result.success and result.fun < best_ssd:
                    best_ssd = result.fun
                    best_params = result.x
            except:
                continue
        
        # Mise à jour des paramètres pour l'échelle suivante
        params[:2] = best_params[:2] / scale  # <-- Conversion inverse en float
        params[2] = best_params[2]  # L'angle ne change pas avec l'échelle
    
    return params

def apply_transformation(img, params):
    """Applique la transformation finale."""
    tx, ty, angle = params
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def calculate_ssd_score(img_ref, img_mov):
    """Calcule le score SSD entre deux images."""
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    return np.sum((img_ref - img_mov)**2)

# Exemple d'utilisation
if __name__ == "__main__":
    img_ref = cv2.imread('images/tisdrin.png')
    img_mov = cv2.imread('images/tisdrin_translated_12_27.jpg')
    
    # Score avant recalage
    ssd_before = calculate_ssd_score(img_ref, img_mov)
    print(f"SSD avant recalage: {ssd_before:.2f}")
    
    # Recalage
    params = register_images_ssd(img_ref, img_mov)
    print(f"Paramètres trouvés - tx: {params[0]:.2f}, ty: {params[1]:.2f}, angle: {params[2]:.2f}°")
    
    # Application de la transformation
    img_registered = apply_transformation(img_mov, params)
    
    # Score après recalage
    ssd_after = calculate_ssd_score(img_ref, img_registered)
    print(f"SSD après recalage: {ssd_after:.2f}")
    print(f"Amélioration: {ssd_before - ssd_after:.2f}")
    
    # Affichage
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    plt.title('Référence')
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB))
    plt.title(f'Mobile (SSD: {ssd_before:.2f})')
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB))
    plt.title(f'Recalée (SSD: {ssd_after:.2f})')
    plt.show()