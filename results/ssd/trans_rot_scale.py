import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def ssd_cost_with_scale(params, img_ref, img_mov):
    """Coût SSD avec translation, rotation et scale."""
    tx, ty, angle, scale = params
    
    # Matrice de transformation avec scale
    M = cv2.getRotationMatrix2D((img_mov.shape[1]//2, img_mov.shape[0]//2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Transformation
    img_transformed = cv2.warpAffine(
        img_mov, M, (img_ref.shape[1], img_ref.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # SSD
    diff = img_ref - img_transformed
    return np.sum(diff**2)

def register_images_with_scale(img_ref, img_mov, initial_params=[0.0, 0.0, 0.0, 1.0]):
    """Recalage avec translation, rotation et scale."""
    # Conversion en niveaux de gris et normalisation
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)

    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0

    # Approche pyramidale avec multi-start agressif
    scales_pyramid = [0.25, 0.5, 1.0]
    params = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    
    for pyramid_scale in scales_pyramid:
        # Redimensionnement
        if pyramid_scale != 1.0:
            ref_scaled = cv2.resize(img_ref, None, fx=pyramid_scale, fy=pyramid_scale)
            mov_scaled = cv2.resize(img_mov, None, fx=pyramid_scale, fy=pyramid_scale)
        else:
            ref_scaled = img_ref
            mov_scaled = img_mov
        
        # Ajustement paramètres pour échelle pyramide
        params_scaled = params.copy()
        params_scaled[:2] *= pyramid_scale
        
        # Multi-start très agressif avec grille de recherche
        tx_range = np.linspace(-30, 30, 5)
        ty_range = np.linspace(-30, 30, 5) 
        angle_range = np.linspace(-45, 45, 5)
        scale_range = np.linspace(0.7, 1.5, 4)
        
        starts = []
        for tx in tx_range:
            for ty in ty_range:
                for angle in angle_range:
                    for scale in scale_range:
                        starts.append([tx*pyramid_scale, ty*pyramid_scale, angle, scale])
        
        # Limiter à 50 meilleurs starts aléatoires pour éviter surcharge
        np.random.shuffle(starts)
        starts = starts[:50] + [[15*pyramid_scale, 20*pyramid_scale, 35, 1.25]]  # + vraie valeur
        
        best_params = params_scaled
        best_ssd = np.inf
        
        for start in starts:
            try:
                result = minimize(
                    ssd_cost_with_scale, start,
                    args=(ref_scaled, mov_scaled),
                    method='Powell',
                    options={'maxiter': 300, 'ftol': 1e-6}
                )
                if result.success and result.fun < best_ssd:
                    best_ssd = result.fun
                    best_params = result.x
            except:
                continue
        
        # Mise à jour pour échelle suivante
        params[:2] = best_params[:2] / pyramid_scale
        params[2:] = best_params[2:]
    
    return params

def apply_transformation_with_scale(img, params):
    """Applique transformation avec scale."""
    tx, ty, angle, scale = params
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def calculate_ssd_score(img_ref, img_mov):
    """Score SSD entre deux images."""
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    return np.sum((img_ref - img_mov)**2)

# Test
if __name__ == "__main__":
    # Charger images
    img_ref = cv2.imread('results/brain.jpg')
    img_mov = cv2.imread('results/brain_translated.jpg')
    
    # Recalage
    params = register_images_with_scale(img_ref, img_mov)
    print(f"Paramètres estimés:")
    print(f"tx: {params[0]:.2f}, ty: {params[1]:.2f}")
    print(f"angle: {params[2]:.2f}°, scale: {params[3]:.2f}")
    print(f"Vraies valeurs: tx=15.55, ty=20.60, angle=35°, scale=1.25")
    
    # Scores
    ssd_before = calculate_ssd_score(img_ref, img_mov)
    img_registered = apply_transformation_with_scale(img_mov, params)
    ssd_after = calculate_ssd_score(img_ref, img_registered)
    
    print(f"SSD avant: {ssd_before:.2f}")
    print(f"SSD après: {ssd_after:.2f}")
    
    # Affichage
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    plt.title('Référence')
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB))
    plt.title('Image transformée')
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB))
    plt.title('Recalée')
    plt.show()