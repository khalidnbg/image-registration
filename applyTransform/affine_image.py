import cv2
import numpy as np

# Charger l'image
image = cv2.imread("images/elephent.png")

# Dimensions de l'image
rows, cols, ch = image.shape

# Définir 3 points de l'image originale
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])

# Définir leurs positions après transformation
pts2 = np.float32([[70, 100], [220, 50], [100, 250]])

# Calculer la matrice de transformation affine
M = cv2.getAffineTransform(pts1, pts2)

# Appliquer la transformation
result = cv2.warpAffine(image, M, (cols, rows))

# Afficher les images
cv2.imshow("Originale", image)
cv2.imshow("Transformée (Affine)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
