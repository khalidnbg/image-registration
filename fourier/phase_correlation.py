import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image
import cv2

def load_image(image_path, grayscale=True):
    """Charge une image depuis un fichier"""
    try:
        img = Image.open(image_path)
        if grayscale:
            img = img.convert('L')
        img = np.array(img, dtype=np.float64)
        print(f"✅ Image chargée: {img.shape}, dtype: {img.dtype}")
        return img
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {image_path}: {e}")
        return None

def phase_correlation_for_personal_images(img1_path, img2_path):
    """
    Corrélation de phase pour tes images personnelles
    
    Parameters:
    img1_path, img2_path: str, chemins vers tes images
    
    Returns:
    results: dict avec tous les résultats
    """
    
    print("🔍 === CORRÉLATION DE PHASE - TES IMAGES ===")
    
    # 1. Chargement des images
    print("\n📂 Chargement des images...")
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    if img1 is None or img2 is None:
        print("❌ Impossible de charger les images")
        return None
    
    # 2. Vérification et redimensionnement
    print(f"Image 1: {img1.shape}")
    print(f"Image 2: {img2.shape}")
    
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (min_w, min_h))
        img2 = cv2.resize(img2, (min_w, min_h))
        print(f"🔄 Images redimensionnées à: {min_h}x{min_w}")
    
    # 3. Prétraitement (padding pour éviter l'aliasing)
    h, w = img1.shape
    new_h, new_w = 2*h - 1, 2*w - 1
    
    img1_pad = np.zeros((new_h, new_w))
    img2_pad = np.zeros((new_h, new_w))
    img1_pad[:h, :w] = img1
    img2_pad[:h, :w] = img2
    
    # Normalisation
    img1_pad = (img1_pad - np.mean(img1_pad)) / (np.std(img1_pad) + 1e-10)
    img2_pad = (img2_pad - np.mean(img2_pad)) / (np.std(img2_pad) + 1e-10)
    
    # 4. ALGORITHME DE CORRÉLATION DE PHASE
    print("\n🧮 Algorithme de corrélation de phase:")
    
    # Étape 1: Calculer les TF de I1 et I2
    print("   1️⃣ Calcul des transformées de Fourier...")
    F1 = fft2(img1_pad)
    F2 = fft2(img2_pad)
    
    # Étape 2: Calculer R(u,v) = F1*conj(F2) / |F1*conj(F2)|
    print("   2️⃣ Calcul du spectre de puissance croisée normalisé...")
    cross_power = F1 * np.conj(F2)
    magnitude = np.abs(cross_power)
    
    # Éviter division par zéro
    epsilon = 1e-10
    magnitude = np.where(magnitude < epsilon, epsilon, magnitude)
    
    # Normalisation par l'amplitude (CŒUR DE LA CORRÉLATION DE PHASE)
    R = cross_power / magnitude
    
    # Étape 3: Calculer la TF inverse
    print("   3️⃣ Transformée de Fourier inverse...")
    phase_corr = np.real(ifft2(R))
    phase_corr = fftshift(phase_corr)
    
    # Étape 4: Rechercher le maximum
    print("   4️⃣ Recherche du pic de corrélation...")
    max_idx = np.unravel_index(np.argmax(phase_corr), phase_corr.shape)
    max_val = phase_corr[max_idx]
    
    # Calculer la translation
    center_y, center_x = np.array(phase_corr.shape) // 2
    ty = max_idx[0] - center_y
    tx = max_idx[1] - center_x
    
    # 5. Calcul de la confiance
    # En corrélation de phase, on évalue la netteté du pic
    std_bg = np.std(phase_corr)
    confidence = max_val / std_bg
    
    # Calculer le SNR (Signal to Noise Ratio)
    mean_bg = np.mean(phase_corr)
    snr = (max_val - mean_bg) / std_bg
    
    # 6. Comparaison avec corrélation classique
    print("\n🔄 Comparaison avec corrélation classique...")
    classic_corr = np.real(ifft2(F1 * np.conj(F2)))
    classic_corr = fftshift(classic_corr)
    
    classic_max_idx = np.unravel_index(np.argmax(classic_corr), classic_corr.shape)
    classic_center = np.array(classic_corr.shape) // 2
    classic_ty = classic_max_idx[0] - classic_center[0]
    classic_tx = classic_max_idx[1] - classic_center[1]
    
    # 7. Résultats
    results = {
        'phase_correlation': phase_corr,
        'classic_correlation': classic_corr,
        'phase_translation': (tx, ty),
        'classic_translation': (classic_tx, classic_ty),
        'confidence': confidence,
        'snr': snr,
        'peak_value': max_val,
        'original_images': (img1, img2)
    }
    
    # 8. Affichage des résultats
    print(f"\n📊 === RÉSULTATS ===")
    print(f"🎯 Corrélation de PHASE:")
    print(f"   Translation détectée: ({tx}, {ty})")
    print(f"   Confiance: {confidence:.2f}")
    print(f"   SNR: {snr:.2f} dB")
    print(f"   Valeur du pic: {max_val:.6f}")
    
    print(f"\n🔵 Corrélation CLASSIQUE (pour comparaison):")
    print(f"   Translation détectée: ({classic_tx}, {classic_ty})")
    print(f"   Valeur max: {np.max(classic_corr):.2e}")
    
    # 9. Interprétation automatique
    print(f"\n🧠 === INTERPRÉTATION ===")
    if abs(tx - classic_tx) <= 1 and abs(ty - classic_ty) <= 1:
        print("✅ Cohérence: Les deux méthodes donnent des résultats similaires")
    else:
        print("⚠️  Différence: Les méthodes donnent des résultats différents")
        print("   → La corrélation de phase peut être plus robuste")
    
    if confidence > 10:
        print("🟢 Confiance ÉLEVÉE - Translation très fiable")
    elif confidence > 5:
        print("🟡 Confiance MODÉRÉE - Translation probable")
    else:
        print("🔴 Confiance FAIBLE - Translation incertaine")
    
    return results

def visualize_phase_correlation_results(results):
    """Visualise les résultats de corrélation de phase"""
    
    if results is None:
        return
    
    phase_corr = results['phase_correlation']
    classic_corr = results['classic_correlation']
    phase_tx, phase_ty = results['phase_translation']
    classic_tx, classic_ty = results['classic_translation']
    img1, img2 = results['original_images']
    
    # Créer la figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ligne 1: Images originales et corrélation de phase
    axes[0,0].imshow(img1, cmap='gray')
    axes[0,0].set_title('Image 1 (Référence)', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(img2, cmap='gray')
    axes[0,1].set_title('Image 2', fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(phase_corr, cmap='hot')
    axes[0,2].set_title(f'Corrélation de Phase\nTranslation: ({phase_tx}, {phase_ty})', 
                       fontsize=12, fontweight='bold', color='red')
    axes[0,2].axis('off')
    
    # Ligne 2: Comparaisons et profils
    axes[1,0].imshow(classic_corr, cmap='hot')
    axes[1,0].set_title(f'Corrélation Classique\nTranslation: ({classic_tx}, {classic_ty})', 
                       fontsize=12, fontweight='bold', color='blue')
    axes[1,0].axis('off')
    
    # Zoom sur le pic de corrélation de phase
    center_y, center_x = np.array(phase_corr.shape) // 2
    crop_size = 80
    y1, y2 = max(0, center_y - crop_size), min(phase_corr.shape[0], center_y + crop_size)
    x1, x2 = max(0, center_x - crop_size), min(phase_corr.shape[1], center_x + crop_size)
    
    phase_crop = phase_corr[y1:y2, x1:x2]
    axes[1,1].imshow(phase_crop, cmap='hot')
    axes[1,1].set_title('Zoom sur le pic\n(Corrélation de Phase)', fontsize=10)
    axes[1,1].axis('off')
    
    # Profils comparatifs
    center_row_phase = phase_corr[center_y + phase_ty, :]
    center_row_classic = classic_corr[center_y + classic_ty, :]
    
    axes[1,2].plot(center_row_phase, 'r-', label='Phase', linewidth=2)
    axes[1,2].plot(center_row_classic/np.max(center_row_classic), 'b--', 
                   label='Classique (normalisée)', alpha=0.7)
    axes[1,2].axvline(center_x + phase_tx, color='red', linestyle=':', alpha=0.8)
    axes[1,2].set_title('Profils de corrélation\n(Coupe horizontale)', fontsize=10)
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Analyse de Corrélation de Phase\nConfiance: {results["confidence"]:.2f} | SNR: {results["snr"]:.2f} dB', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# FONCTION PRINCIPALE POUR TES IMAGES
def analyze_my_images(path1, path2):
    """
    Fonction principale pour analyser tes images avec corrélation de phase
    
    Usage:
    analyze_my_images("mon_image1.jpg", "mon_image2.jpg")
    """
    # Analyser avec corrélation de phase
    results = phase_correlation_for_personal_images(path1, path2)
    
    if results is not None:
        # Visualiser les résultats
        visualize_phase_correlation_results(results)
        
        # Retourner pour usage ultérieur
        return results
    else:
        print("❌ Échec de l'analyse")
        return None

# USAGE DIRECT
if __name__ == "__main__":
    print("🎯 CORRÉLATION DE PHASE POUR TES IMAGES")
    print("=" * 50)
    
    # ⚠️ CHANGE CES CHEMINS VERS TES VRAIES IMAGES ⚠️
    image1_path = "images/img_org.png"  # ← TON CHEMIN ICI
    image2_path = "images/img_org_translated_35_48.png"  # ← TON CHEMIN ICI
    
    try:
        results = analyze_my_images(image1_path, image2_path)
        
        if results:
            print(f"\n🎉 Analyse terminée avec succès!")
            print(f"📈 Translation finale: {results['phase_translation']}")
            
    except FileNotFoundError:
        print("❌ Images non trouvées!")
        print("💡 Vérifie les chemins:")
        print(f"   - {image1_path}")
        print(f"   - {image2_path}")
        print("\n🔧 Change les chemins dans le code à la ligne 'image1_path = ...'")
    except Exception as e:
        print(f"❌ Erreur: {e}")