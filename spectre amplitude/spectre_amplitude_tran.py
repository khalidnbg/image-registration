import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.ndimage import affine_transform
import matplotlib.gridspec as gridspec

# === Paramètres ===
image_path = "images/img_org_translated_img.PNG"  # Remplace avec ton image réelle
x_translation = 20
y_translation = 15
alpha_deg = 0
lambda_scale = 1
shift = (x_translation, y_translation)
alpha = np.deg2rad(alpha_deg)

# === Charger image ===
image_I = Image.open(image_path).convert("L")
image_np = np.array(image_I)

# === Matrice de transformation (affine) ===
# 1. Mise à l’échelle
S = np.array([
    [1/lambda_scale, 0],
    [0, 1/lambda_scale]
])

# 2. Rotation
R = np.array([
    [np.cos(alpha), -np.sin(alpha)],
    [np.sin(alpha),  np.cos(alpha)]
])

# 3. Matrice finale A = S @ R
A = S @ R

# 4. Translation compensée pour centrage
center = np.array(image_np.shape[::-1]) / 2
offset = center - A @ center + [x_translation, y_translation]

# === Appliquer la transformation ===
image_J_np = affine_transform(image_np, A, offset=offset, order=1)

# === Affichage ===
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2)

# Image I
ax1 = plt.subplot(gs[0, 0])
ax1.imshow(image_np, cmap='gray')
ax1.set_title("Image de référence (I)")
ax1.axis('off')

# Image J
ax2 = plt.subplot(gs[0, 1])
ax2.imshow(image_J_np, cmap='gray')
ax2.set_title(f"Image transformée (J)\nShift {shift}, α={alpha_deg}°, λ={lambda_scale}")
ax2.axis('off')

plt.tight_layout()
plt.show()
