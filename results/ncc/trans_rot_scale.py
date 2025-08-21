import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def ncc_cost_with_scale(params, img_ref, img_mov):
    """Coût NCC avec translation, rotation et scale."""
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
    
    # Calcul NCC
    img_ref_flat = img_ref.flatten()
    img_mov_flat = img_transformed.flatten()
    
    mean_ref = np.mean(img_ref_flat)
    mean_mov = np.mean(img_mov_flat)
    
    numerator = np.sum((img_ref_flat - mean_ref) * (img_mov_flat - mean_mov))
    denominator = np.sqrt(
        np.sum((img_ref_flat - mean_ref) ** 2) * 
        np.sum((img_mov_flat - mean_mov) ** 2)
    )
    
    if denominator == 0:
        return 1.0
    
    ncc = numerator / denominator
    return -ncc  # Minimiser pour maximiser NCC

def register_images_ncc_with_scale(img_ref, img_mov, initial_params=[0.0, 0.0, 0.0, 1.0]):
    """Recalage NCC rapide avec recherche hiérarchique."""
    # Conversion en niveaux de gris et normalisation
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0

    # Étape 1: Recherche grossière sur image réduite (très rapide)
    ref_small = cv2.resize(img_ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    mov_small = cv2.resize(img_mov, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    
    # Grille grossière 
    best_params = initial_params
    best_ncc = -np.inf
    
    for tx in [-20, -10, 0, 10, 20]:
        for ty in [-20, -10, 0, 10, 20]:
            for angle in [-30, 0, 30]:
                for scale in [0.8, 1.0, 1.2, 1.4]:
                    params_test = [tx*0.25, ty*0.25, angle, scale]
                    ncc_val = -ncc_cost_with_scale(params_test, ref_small, mov_small)
                    if ncc_val > best_ncc:
                        best_ncc = ncc_val
                        best_params = [tx, ty, angle, scale]
    
    print(f"Recherche grossière: tx={best_params[0]:.1f}, ty={best_params[1]:.1f}, "
          f"angle={best_params[2]:.1f}°, scale={best_params[3]:.2f}")
    
    # Étape 2: Raffinement local avec multi-start restreint
    starts = [
        best_params,
        [best_params[0]+5, best_params[1]+5, best_params[2]+10, best_params[3]+0.1],
        [best_params[0]-5, best_params[1]-5, best_params[2]-10, best_params[3]-0.1],
        [best_params[0]+10, best_params[1], best_params[2]+5, best_params[3]+0.2],
        [best_params[0], best_params[1]+10, best_params[2]-5, best_params[3]-0.2]
    ]
    
    final_best = best_params
    final_best_ncc = -np.inf
    
    for start in starts:
        try:
            result = minimize(
                ncc_cost_with_scale, start,
                args=(img_ref, img_mov),
                method='Powell',
                options={'maxiter': 300, 'ftol': 1e-6}
            )
            current_ncc = -result.fun
            if result.success and current_ncc > final_best_ncc:
                final_best_ncc = current_ncc
                final_best = result.x
        except:
            continue
    
    return final_best

def apply_transformation_with_scale(img, params):
    """Applique transformation avec scale."""
    tx, ty, angle, scale = params
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def calculate_ncc_score(img_ref, img_mov):
    """Score NCC entre deux images."""
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    return -ncc_cost_with_scale([0.0, 0.0, 0.0, 1.0], img_ref, img_mov)

# Test
if __name__ == "__main__":
    # Charger images
    img_ref = cv2.imread('results/brain.jpg')
    img_mov = cv2.imread('results/brain_transformed.jpg')
    
    # Recalage
    params = register_images_ncc_with_scale(img_ref, img_mov)
    print(f"Paramètres estimés (NCC):")
    print(f"tx: {params[0]:.2f}, ty: {params[1]:.2f}")
    print(f"angle: {params[2]:.2f}°, scale: {params[3]:.2f}")
    print(f"Vraies valeurs: tx=15.55, ty=20.60, angle=35°, scale=1.25")
    
    # Scores
    ncc_before = calculate_ncc_score(img_ref, img_mov)
    img_registered = apply_transformation_with_scale(img_mov, params)
    ncc_after = calculate_ncc_score(img_ref, img_registered)
    
    print(f"NCC avant: {ncc_before:.4f}")
    print(f"NCC après: {ncc_after:.4f}")
    print(f"Amélioration: {ncc_after - ncc_before:.4f}")
    
    # Affichage
    plt.figure(figsize=(12, 4))
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