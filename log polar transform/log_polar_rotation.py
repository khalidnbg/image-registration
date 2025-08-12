import numpy as np
import matplotlib.pyplot as plt

class SimpleLogPolar:
    
    def load_image(self, image_path):
        """Charge une image"""
        img = plt.imread(image_path)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)  # Convertir en gris
        return img.astype(np.float64)
    
    def to_log_polar(self, image):
        """Convertit image cartésienne vers log-polaire"""
        h, w = image.shape
        center = (w//2, h//2)
        
        # Taille image log-polaire
        lp_height, lp_width = 150, 360
        log_polar = np.zeros((lp_height, lp_width))
        
        max_r = min(center)
        
        for i in range(lp_height):
            for j in range(lp_width):
                # Angle (0 à 2π)
                theta = (j / lp_width) * 2 * np.pi
                
                # Rayon (échelle log)
                r = 1 + (max_r-1) * (i / lp_height)
                
                # Coordonnées cartésiennes
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                
                # Échantillonnage
                if 0 <= x < w-1 and 0 <= y < h-1:
                    log_polar[i, j] = image[int(y), int(x)]
        
        return log_polar
    
    def detect_rotation(self, img1, img2):
        """Détecte rotation entre deux images"""
        lp1 = self.to_log_polar(img1)
        lp2 = self.to_log_polar(img2)
        
        # Profils angulaires
        profile1 = np.mean(lp1, axis=0)
        profile2 = np.mean(lp2, axis=0)
        
        # Corrélation croisée
        corr = np.correlate(np.tile(profile1, 2), profile2, mode='valid')
        shift = np.argmax(corr)
        
        # Conversion en degrés
        angle = (shift / len(profile1)) * 360
        if angle > 180:
            angle -= 360
            
        return angle, lp1, lp2
    
    def show_results(self, img1, img2, angle, lp1, lp2):
        """Affiche les résultats"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Images originales
        axes[0,0].imshow(img1, cmap='gray')
        axes[0,0].set_title('Image 1')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(img2, cmap='gray')
        axes[0,1].set_title('Image 2')
        axes[0,1].axis('off')
        
        # Images log-polaires
        axes[1,0].imshow(lp1, cmap='gray', aspect='auto')
        axes[1,0].set_title('Log-polaire 1')
        axes[1,0].set_ylabel('log(r)')
        
        axes[1,1].imshow(lp2, cmap='gray', aspect='auto')
        axes[1,1].set_title(f'Log-polaire 2\nRotation: {angle:.1f}°')
        axes[1,1].set_xlabel('Angle θ')
        
        plt.tight_layout()
        plt.show()
        
        return angle

# Utilisation simple
if __name__ == "__main__":
    lp = SimpleLogPolar()
    
    # REMPLACEZ CES CHEMINS PAR VOS IMAGES
    img1_path = "images/tisdrin.png"  # Votre image 1
    img2_path = "images/tisdrin_rotated_18.jpg"  # Votre image 2
    
    try:
        # Charger images
        img1 = lp.load_image(img1_path)
        img2 = lp.load_image(img2_path)
        
        print(f"Images chargées: {img1.shape}, {img2.shape}")
        
        # Détecter rotation
        angle, lp1, lp2 = lp.detect_rotation(img1, img2)
        
        # Afficher résultats
        lp.show_results(img1, img2, angle, lp1, lp2)
        
        print(f"Rotation détectée: {angle:.2f}°")
        
    except Exception as e:
        print(f"Erreur: {e}")
        print("Vérifiez les chemins d'images")

# INSTRUCTIONS:
# 1. Remplacez "image1.jpg" et "image2.jpg" par vos vraies images
# 2. Lancez le script
# 3. La rotation sera affichée en degrés