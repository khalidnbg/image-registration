import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float

# ===== CHARGEMENT DE VOTRE IMAGE =====
img_ref = cv2.imread('results/brain.jpg')
if img_ref is None:
    raise FileNotFoundError("❌ Impossible de charger l'image 'results/brain.jpg'. Vérifiez le chemin.")

# Conversion en niveaux de gris si l'image est en couleur
if len(img_ref.shape) == 3:
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

# Normalisation entre 0 et 1
image = img_as_float(img_ref)

print(f"Image chargée: {image.shape}")
print(f"Type d'image: {'Niveaux de gris' if image.ndim == 2 else 'Couleur'}")

# ===== ROTATIONS À TESTER =====
rotations_reelles = [15, 35, -15.5, 40, -20]
rotations_estimées = []
erreurs = []

# Rayon pour la transformation polaire (ajusté pour votre image)
radius = min(image.shape) // 2

# ===== TRAITEMENT POUR CHAQUE ROTATION =====
for i, angle in enumerate(rotations_reelles):
    print(f"\n{'='*50}")
    print(f"TEST {i+1}: Rotation de {angle}°")
    print(f"{'='*50}")
    
    # Application de la rotation
    rotated = rotate(image, angle)
    
    # Transformation polaire
    if image.ndim == 3:
        image_polar = warp_polar(image, radius=radius, channel_axis=-1)
        rotated_polar = warp_polar(rotated, radius=radius, channel_axis=-1)
    else:
        image_polar = warp_polar(image, radius=radius)
        rotated_polar = warp_polar(rotated, radius=radius)
    
    # Détection de la rotation par corrélation de phase
    shifts, error, phasediff = phase_cross_correlation(
        image_polar, rotated_polar, normalization=None
    )
    
    angle_estime = shifts[0]
    rotations_estimées.append(angle_estime)
    erreur = abs(angle_estime - angle)
    erreurs.append(erreur)
    
    print(f'Angle appliqué: {angle}°')
    print(f'Angle détecté: {angle_estime:.2f}°')
    print(f'Erreur: {error:.6f}')
    print(f'Différence absolue: {erreur:.2f}°')
    
    # ===== AFFICHAGE DES RÉSULTATS POUR CETTE ROTATION =====
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes.ravel()
    
    cmap = 'gray' if image.ndim == 2 else None
    
    ax[0].set_title("Image Originale")
    ax[0].imshow(image, cmap=cmap)
    ax[0].axis('off')
    
    ax[1].set_title(f"Image Tournée ({angle}°)")
    ax[1].imshow(rotated, cmap=cmap)
    ax[1].axis('off')
    
    ax[2].set_title("Transformation Polaire - Original")
    ax[2].imshow(image_polar, cmap=cmap)
    ax[2].axis('off')
    
    ax[3].set_title("Transformation Polaire - Tournée")
    ax[3].imshow(rotated_polar, cmap=cmap)
    ax[3].axis('off')
    
    plt.suptitle(f'Test {i+1}: Rotation de {angle}° → Détectée: {angle_estime:.1f}° (Erreur: {erreur:.2f}°)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ===== RÉSULTATS FINAUX =====
print(f"\n{'='*60}")
print("RÉSULTATS FINAUX - CORRÉLATION DE PHASE")
print(f"{'='*60}")

for i, (angle_reel, angle_est, erreur) in enumerate(zip(rotations_reelles, rotations_estimées, erreurs)):
    print(f"Test {i+1}: {angle_reel:6.1f}° → {angle_est:6.1f}° | Erreur: {erreur:5.2f}°")

print(f"\n{'='*60}")
print("STATISTIQUES:")
print(f"{'='*60}")

erreur_moyenne = np.mean(erreurs)
erreur_max = np.max(erreurs)
erreur_min = np.min(erreurs)

print(f"Erreur moyenne: {erreur_moyenne:.2f}°")
print(f"Erreur maximale: {erreur_max:.2f}°")
print(f"Erreur minimale: {erreur_min:.2f}°")
print(f"Précision moyenne: {100 - erreur_moyenne:.1f}%")

# ===== GRAPHIQUE DE COMPARAISON =====
plt.figure(figsize=(10, 6))
plt.plot(rotations_reelles, rotations_estimées, 'bo-', markersize=8, 
         label='Rotations estimées', linewidth=2)
plt.plot(rotations_reelles, rotations_reelles, 'r--', 
         label='Rotations réelles', linewidth=2)

plt.xlabel('Rotation appliquée (°)')
plt.ylabel('Rotation estimée (°)')
plt.title('Comparaison des rotations estimées vs réelles\n(Corrélation de Phase)')
plt.legend()
plt.grid(True, alpha=0.3)

# Ajouter les valeurs d'erreur
for i, (x, y, err) in enumerate(zip(rotations_reelles, rotations_estimées, erreurs)):
    plt.annotate(f'err: {err:.1f}°', (x, y), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color='red')

plt.tight_layout()
plt.show()

# ===== TABLEAU RÉCAPITULATIF =====
print(f"\n{'='*80}")
print("TABLEAU RÉCAPITULATIF")
print(f"{'='*80}")
print(f"{'Test':^6} | {'Rotation réelle':^15} | {'Rotation estimée':^15} | {'Erreur':^10} | {'Statut':^15}")
print(f"{'-'*80}")

for i, (reel, estime, erreur) in enumerate(zip(rotations_reelles, rotations_estimées, erreurs)):
    if erreur < 1.0:
        statut = "✓ EXCELLENT"
    elif erreur < 3.0:
        statut = "✓ TRÈS BON"
    elif erreur < 5.0:
        statut = "✓ BON"
    else:
        statut = "⚠ MOYEN"
    
    print(f"{i+1:^6} | {reel:^15.1f} | {estime:^15.1f} | {erreur:^10.2f} | {statut:^15}")

print(f"{'-'*80}")