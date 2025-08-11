import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image
import cv2

def load_image(image_path, grayscale=True):
    """
    Charge une image depuis un fichier
    
    Parameters:
    image_path: str, chemin vers l'image
    grayscale: bool, convertir en niveaux de gris
    
    Returns:
    img: array 2D ou 3D
    """
    try:
        # Méthode 1: avec PIL
        img = Image.open(image_path)
        if grayscale:
            img = img.convert('L')
        img = np.array(img)
        
        # Alternative avec OpenCV (décommente si préféré)
        # img = cv2.imread(image_path)
        # if grayscale and len(img.shape) == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img
        
    except Exception as e:
        print(f"Erreur lors du chargement de {image_path}: {e}")
        return None

def preprocess_images(img1, img2, resize=True, normalize=True):
    """
    Prétraitement des images pour la corrélation
    
    Parameters:
    img1, img2: arrays, les deux images
    resize: bool, redimensionner à la même taille
    normalize: bool, normaliser les valeurs
    
    Returns:
    img1_proc, img2_proc: images prétraitées
    """
    
    # Copie pour éviter de modifier les originales
    img1_proc = img1.copy().astype(np.float64)
    img2_proc = img2.copy().astype(np.float64)
    
    # Redimensionnement si nécessaire
    if resize and img1.shape != img2.shape:
        # Prendre la taille minimale
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        
        img1_proc = cv2.resize(img1_proc, (min_w, min_h))
        img2_proc = cv2.resize(img2_proc, (min_w, min_h))
        
        print(f"Images redimensionnées à: {min_h}x{min_w}")
    
    # Normalisation
    if normalize:
        img1_proc = (img1_proc - np.mean(img1_proc)) / np.std(img1_proc)
        img2_proc = (img2_proc - np.mean(img2_proc)) / np.std(img2_proc)
    
    return img1_proc, img2_proc

def correlation_fft_custom(img1_path, img2_path, show_results=True):
    """
    Corrélation FFT complète pour images personnelles
    
    Parameters:
    img1_path, img2_path: str, chemins vers les images
    show_results: bool, afficher les résultats
    
    Returns:
    correlation: matrice de corrélation
    translation: tuple (tx, ty)
    confidence: float, confiance du résultat
    """
    
    # 1. Charger les images
    print("Chargement des images...")
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    if img1 is None or img2 is None:
        return None, None, 0
    
    print(f"Image 1: {img1.shape}, dtype: {img1.dtype}")
    print(f"Image 2: {img2.shape}, dtype: {img2.dtype}")
    
    # 2. Prétraitement
    img1_proc, img2_proc = preprocess_images(img1, img2)
    
    # 3. Padding pour éviter l'aliasing
    h1, w1 = img1_proc.shape
    h2, w2 = img2_proc.shape
    
    new_h = h1 + h2 - 1
    new_w = w1 + w2 - 1
    
    # Créer les images paddées
    img1_pad = np.zeros((new_h, new_w))
    img2_pad = np.zeros((new_h, new_w))
    
    img1_pad[:h1, :w1] = img1_proc
    img2_pad[:h2, :w2] = img2_proc
    
    # 4. Calcul FFT
    print("Calcul des transformées de Fourier...")
    F1 = fft2(img1_pad)
    F2 = fft2(img2_pad)
    
    # 5. Corrélation dans le domaine fréquentiel
    C = F1 * np.conj(F2)
    
    # 6. Retour au domaine spatial
    correlation = np.real(ifft2(C))
    correlation = fftshift(correlation)
    
    # 7. Trouver le pic principal
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    max_val = correlation[max_idx]
    
    # 8. Calculer la translation
    center_y, center_x = np.array(correlation.shape) // 2
    ty = max_idx[0] - center_y
    tx = max_idx[1] - center_x
    
    # 9. Calculer la confiance (ratio pic/moyenne)
    confidence = max_val / np.mean(correlation)
    
    print(f"\nRésultats:")
    print(f"Translation détectée: tx={tx}, ty={ty}")
    print(f"Confiance: {confidence:.2f}")
    
    # 10. Affichage des résultats
    if show_results:
        visualize_results(img1, img2, correlation, (tx, ty), confidence)
    
    return correlation, (tx, ty), confidence

def visualize_results(img1, img2, correlation, translation, confidence):
    """Visualise les résultats de la corrélation"""
    
    tx, ty = translation
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Image 1
    axes[0,0].imshow(img1, cmap='gray')
    axes[0,0].set_title('Image 1 (référence)')
    axes[0,0].axis('off')
    
    # Image 2  
    axes[0,1].imshow(img2, cmap='gray')
    axes[0,1].set_title('Image 2')
    axes[0,1].axis('off')
    
    # Corrélation complète
    axes[0,2].imshow(correlation, cmap='hot')
    axes[0,2].set_title('Corrélation croisée')
    axes[0,2].axis('off')
    
    # Zoom sur le pic
    center_y, center_x = np.array(correlation.shape) // 2
    crop_size = 100
    y1, y2 = max(0, center_y - crop_size), min(correlation.shape[0], center_y + crop_size)
    x1, x2 = max(0, center_x - crop_size), min(correlation.shape[1], center_x + crop_size)
    
    axes[1,0].imshow(correlation[y1:y2, x1:x2], cmap='hot')
    axes[1,0].set_title(f'Pic: ({tx}, {ty})')
    axes[1,0].axis('off')
    
    # Profil de corrélation (coupe horizontale)
    center_row = correlation[center_y + ty, :]
    axes[1,1].plot(center_row)
    axes[1,1].axvline(center_x + tx, color='red', linestyle='--', label='Pic détecté')
    axes[1,1].set_title('Profil horizontal')
    axes[1,1].legend()
    
    # Profil de corrélation (coupe verticale)
    center_col = correlation[:, center_x + tx]
    axes[1,2].plot(center_col)
    axes[1,2].axhline(center_y + ty, color='red', linestyle='--', label='Pic détecté')
    axes[1,2].set_title('Profil vertical')
    axes[1,2].legend()
    
    plt.suptitle(f'Corrélation FFT - Confiance: {confidence:.2f}')
    plt.tight_layout()
    plt.show()

# Fonction simple d'utilisation
def quick_correlation(path1, path2):
    """
    Version rapide pour usage direct
    
    Usage:
    correlation, (tx, ty), confidence = quick_correlation("img1.jpg", "img2.jpg")
    """
    return correlation_fft_custom(path1, path2, show_results=True)

# Exemple d'usage
if __name__ == "__main__":
    # Remplace par tes vrais chemins d'images
    image1_path = "images/tisdrin.png"  # ← Change ici
    image2_path = "images/.jpg"  # ← Change ici
    
    print("=== Corrélation FFT pour images personnelles ===")
    
    try:
        # Lancer la corrélation
        correlation, translation, confidence = quick_correlation(image1_path, image2_path)
        
        if correlation is not None:
            tx, ty = translation
            print(f"\n✅ Succès!")
            print(f"📍 Décalage détecté: x={tx}, y={ty}")
            print(f"🎯 Confiance: {confidence:.2f}")
            
            # Interpréter la confiance
            if confidence > 5:
                print("🟢 Corrélation très forte - Translation fiable")
            elif confidence > 2:
                print("🟡 Corrélation modérée - Translation possible")
            else:
                print("🔴 Corrélation faible - Translation incertaine")
        else:
            print("❌ Échec du traitement")
            
    except FileNotFoundError:
        print("❌ Fichiers d'images non trouvés!")
        print("💡 Assure-toi que les chemins sont corrects:")
        print(f"   - {image1_path}")
        print(f"   - {image2_path}")