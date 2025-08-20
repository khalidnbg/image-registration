import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Fonctions SSD et recalage avec rotation centrée corrigée ---
def ssd_cost(params, img_ref, img_mov):
    tx, ty, angle = params
    # Centre de l'image
    center = (img_mov.shape[1] / 2.0, img_mov.shape[0] / 2.0)
    
    # Matrice de rotation centrée
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Appliquer la translation après la rotation
    M[0, 2] += tx
    M[1, 2] += ty
    
    img_transformed = cv2.warpAffine(img_mov, M, (img_ref.shape[1], img_ref.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    diff = img_ref - img_transformed
    return np.sum(diff**2)

def register_images_ssd(img_ref, img_mov, initial_params=[0.0, 0.0, 0.0]):
    if len(img_ref.shape) == 3: img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3: img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32)/255.0
    img_mov = img_mov.astype(np.float32)/255.0

    params = np.array(initial_params, dtype=np.float64)
    starts = [
        params,
        params + [0.0, 0.0, 5.0],
        params + [0.0, 0.0, -5.0],
        params + [0.0, 0.0, 10.0],
        params + [0.0, 0.0, -10.0]
    ]

    best_params = params
    best_ssd = np.inf
    for start in starts:
        try:
            result = minimize(ssd_cost, start, args=(img_ref, img_mov), method='Powell', options={'maxiter':500, 'ftol':1e-6})
            if result.success and result.fun < best_ssd:
                best_ssd = result.fun
                best_params = result.x
        except:
            continue
    return best_params

def apply_transformation(img, params):
    tx, ty, angle = params
    # Centre de l'image
    center = (img.shape[1] / 2.0, img.shape[0] / 2.0)
    
    # Matrice de rotation centrée
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Appliquer la translation après la rotation
    M[0, 2] += tx
    M[1, 2] += ty
    
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

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

# --- Test 5 rotations différentes avec rotation centrée ---
if __name__ == "__main__":
    # Charger l'image
    img_ref = cv2.imread('results/brain.jpg')
    if img_ref is None:
        # Créer une image de test si l'image n'est pas trouvée
        print("Image non trouvée, création d'une image de test...")
        img_ref = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(img_ref, (50, 50), (200, 200), (255, 0, 0), -1)
        cv2.circle(img_ref, (128, 128), 40, (0, 255, 0), -1)
    
    rotations_reelles = [15, 35, -15.50, 40, -20]  # rotations appliquées
    rotations_estimées = []
    erreurs = []

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

        # Recalage
        params_est = register_images_ssd(img_ref, img_mov)
        rotation_estimee = params_est[2]
        rotations_estimées.append(rotation_estimee)
        erreur = abs(rotation_estimee - angle)
        erreurs.append(erreur)
        
        print(f"Rotation réelle: {angle:6.2f}°, estimée: {rotation_estimee:6.2f}°, erreur: {erreur:6.2f}°")

    plt.tight_layout()
    plt.show()

    # --- Affichage de la comparaison ---
    plt.figure(figsize=(12, 5))
    
    # Graphique 1: Comparaison rotations réelles vs estimées
    plt.subplot(1, 2, 1)
    plt.plot(rotations_reelles, rotations_estimées, 'o-', color='blue', label='Rotation estimée', markersize=8, linewidth=2)
    plt.plot(rotations_reelles, rotations_reelles, 'r--', label='Rotation réelle', linewidth=2)
    plt.xlabel("Rotation appliquée (°)")
    plt.ylabel("Rotation estimée (°)")
    plt.title("Comparaison des rotations estimées vs réelles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Erreurs
    plt.subplot(1, 2, 2)
    plt.bar(range(len(rotations_reelles)), erreurs, color='orange', alpha=0.7)
    plt.xlabel("Test #")
    plt.ylabel("Erreur absolue (°)")
    plt.title("Erreurs d'estimation de rotation")
    plt.xticks(range(len(rotations_reelles)), [f'Test {i+1}' for i in range(len(rotations_reelles))])
    plt.grid(True, alpha=0.3)
    
    # Ajouter les valeurs d'erreur sur les barres
    for i, erreur in enumerate(erreurs):
        plt.text(i, erreur + 0.1, f'{erreur:.2f}°', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

    # --- Statistiques de performance ---
    erreur_moyenne = np.mean(erreurs)
    erreur_max = np.max(erreurs)
    
    print(f"\n--- Statistiques de performance ---")
    print(f"Erreur moyenne: {erreur_moyenne:.4f}°")
    print(f"Erreur maximale: {erreur_max:.4f}°")
    print(f"Précision moyenne: {100 - erreur_moyenne:.2f}%")