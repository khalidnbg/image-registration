import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def ncc_cost(params, img_ref, img_mov, mask=None):
    """Coût NCC (à minimiser car on maximise NCC)."""
    tx, ty, angle = params
    
    # Centre de l'image
    center = (img_mov.shape[1] / 2.0, img_mov.shape[0] / 2.0)
    
    # Matrice de transformation avec rotation centrée
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
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
    """Applique la transformation finale avec rotation centrée."""
    tx, ty, angle = params
    # Centre de l'image
    center = (img.shape[1] / 2.0, img.shape[0] / 2.0)
    
    # Matrice de transformation avec rotation centrée
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
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

def create_rotated_image_with_centered_rotation(img, angle_degrees):
    """
    Crée une image tournée avec rotation centrée correcte
    """
    # Centre de l'image
    center = (img.shape[1] / 2.0, img.shape[0] / 2.0)
    
    # Matrice de rotation
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Appliquer la rotation sans translation supplémentaire
    img_rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return img_rotated

# --- Test 5 rotations différentes avec NCC ---
if __name__ == "__main__":
    # Charger l'image
    img_ref = cv2.imread('results/brain.jpg')
    if img_ref is None:
        # Créer une image de test si l'image n'est pas trouvée
        print("Image non trouvée, création d'une image de test...")
        img_ref = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(img_ref, (50, 50), (200, 200), (255, 0, 0), -1)
        cv2.circle(img_ref, (128, 128), 40, (0, 255, 0), -1)
        cv2.putText(img_ref, 'TEST', (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    rotations_reelles = [15, 35, -15.50, 40, -20]  # rotations appliquées
    rotations_estimées = []
    erreurs = []
    ncc_scores = []

    # Afficher les images originales et transformées pour vérification
    plt.figure(figsize=(15, 10))
    
    for i, angle in enumerate(rotations_reelles):
        # Créer l'image mobile par rotation CENTRÉE
        img_mov = create_rotated_image_with_centered_rotation(img_ref, angle)
        
        # Afficher les images pour vérification visuelle
        plt.subplot(2, len(rotations_reelles), i + 1)
        plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
        plt.title(f'Référence\nRotation: 0°')
        plt.axis('off')
        
        plt.subplot(2, len(rotations_reelles), i + 1 + len(rotations_reelles))
        plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB))
        plt.title(f'Rotatée\nRotation: {angle}°')
        plt.axis('off')

        # Calcul du score NCC avant recalage
        ncc_avant = calculate_ncc_score(img_ref, img_mov)
        
        # Recalage avec NCC
        params_est = register_images_ncc(img_ref, img_mov)
        rotation_estimee = params_est[2]
        rotations_estimées.append(rotation_estimee)
        erreur = abs(rotation_estimee - angle)
        erreurs.append(erreur)
        
        # Calcul du score NCC après recalage
        img_recalee = apply_transformation(img_mov, params_est)
        ncc_apres = calculate_ncc_score(img_ref, img_recalee)
        ncc_scores.append((ncc_avant, ncc_apres))
        
        print(f"Rotation réelle: {angle:6.2f}°, estimée: {rotation_estimee:6.2f}°, erreur: {erreur:6.2f}°")
        print(f"NCC avant: {ncc_avant:.4f}, après: {ncc_apres:.4f}, gain: {ncc_apres - ncc_avant:+.4f}")
        print("-" * 50)

    plt.tight_layout()
    plt.show()

    # --- Affichage de la comparaison ---
    plt.figure(figsize=(15, 5))
    
    # Graphique 1: Comparaison rotations réelles vs estimées
    plt.subplot(1, 3, 1)
    plt.plot(rotations_reelles, rotations_estimées, 'o-', color='blue', label='Rotation estimée', markersize=8, linewidth=2)
    plt.plot(rotations_reelles, rotations_reelles, 'r--', label='Rotation réelle', linewidth=2)
    plt.xlabel("Rotation appliquée (°)")
    plt.ylabel("Rotation estimée (°)")
    plt.title("Comparaison des rotations estimées vs réelles (NCC)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Erreurs
    plt.subplot(1, 3, 2)
    plt.bar(range(len(rotations_reelles)), erreurs, color='orange', alpha=0.7)
    plt.xlabel("Test #")
    plt.ylabel("Erreur absolue (°)")
    plt.title("Erreurs d'estimation de rotation (NCC)")
    plt.xticks(range(len(rotations_reelles)), [f'Test {i+1}' for i in range(len(rotations_reelles))])
    plt.grid(True, alpha=0.3)
    
    # Ajouter les valeurs d'erreur sur les barres
    for i, erreur in enumerate(erreurs):
        plt.text(i, erreur + 0.1, f'{erreur:.2f}°', ha='center', va='bottom')
    
    # Graphique 3: Scores NCC
    plt.subplot(1, 3, 3)
    ncc_avants = [score[0] for score in ncc_scores]
    ncc_apres = [score[1] for score in ncc_scores]
    x_pos = np.arange(len(rotations_reelles))
    width = 0.35
    
    plt.bar(x_pos - width/2, ncc_avants, width, label='NCC avant', alpha=0.7, color='red')
    plt.bar(x_pos + width/2, ncc_apres, width, label='NCC après', alpha=0.7, color='green')
    plt.xlabel("Test #")
    plt.ylabel("Score NCC")
    plt.title("Scores NCC avant/après recalage")
    plt.xticks(x_pos, [f'Test {i+1}' for i in range(len(rotations_reelles))])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # --- Statistiques de performance ---
    erreur_moyenne = np.mean(erreurs)
    erreur_max = np.max(erreurs)
    gains_ncc = [apres - avant for avant, apres in ncc_scores]
    gain_moyen = np.mean(gains_ncc)
    
    print(f"\n--- Statistiques de performance NCC ---")
    print(f"Erreur moyenne de rotation: {erreur_moyenne:.4f}°")
    print(f"Erreur maximale de rotation: {erreur_max:.4f}°")
    print(f"Gain NCC moyen: {gain_moyen:+.4f}")
    print(f"Précision moyenne: {100 - erreur_moyenne:.2f}%")
    
    # Tableau récapitulatif
    print(f"\n{'Test':^6} | {'Rotation réelle':^15} | {'Rotation estimée':^15} | {'Erreur':^10} | {'NCC avant':^10} | {'NCC après':^10} | {'Gain NCC':^10}")
    print("-" * 85)
    for i, (angle_reel, angle_est, erreur, (ncc_av, ncc_ap)) in enumerate(zip(rotations_reelles, rotations_estimées, erreurs, ncc_scores)):
        print(f"{i+1:^6} | {angle_reel:^15.2f} | {angle_est:^15.2f} | {erreur:^10.2f} | {ncc_av:^10.4f} | {ncc_ap:^10.4f} | {ncc_ap-ncc_av:^+10.4f}")