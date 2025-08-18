import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

# Charger l'image "camera" depuis scikit-learn
from skimage import data
img = data.camera()

# Convertir en uint8 si nécessaire
img = np.array(img, dtype=np.uint8)

# Créer une seconde image transformée (ex: rotation légère)
rows, cols = img.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)  # rotation de 15°
img2 = cv2.warpAffine(img, M, (cols, rows))

# Initialiser SIFT
sift = cv2.SIFT_create()

# Détection et description des points clés
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Utilisation du matcher FLANN (rapide pour grands descripteurs comme SIFT)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # nb de vérifications
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Trouver les 2 plus proches voisins
matches = flann.knnMatch(des1, des2, k=2)

# Appliquer le ratio test de Lowe (0.8)
good_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(m)

print(f"Nombre de correspondances brutes : {len(matches)}")
print(f"Nombre de bonnes correspondances (après ratio test) : {len(good_matches)}")

# Dessiner les correspondances retenues
img_matches = cv2.drawMatches(img, kp1, img2, kp2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12,6))
plt.imshow(img_matches, cmap='gray')
plt.title("Mise en correspondance des points clés (SIFT + Ratio Test)")
plt.axis('off')
plt.show()
