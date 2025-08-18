from PIL import Image

# Charger l'image
img = Image.open(r"C:\Users\khalid\projects\image registration\code\applyTransform\number2.png")

# Facteur d'échelle (70% de la taille originale)
scale = 0.7

# Nouvelle taille calculée
new_size = (int(img.width * scale), int(img.height * scale))

# Garder la résolution DPI d'origine (par défaut 300 si non défini)
dpi = img.info.get('dpi', (300, 300))

# Redimensionner avec interpolation haute qualité
resized_img = img.resize(new_size, Image.LANCZOS)

# Sauvegarder avec la même résolution
resized_img.save("num2.jpg", dpi=dpi)

print(f"✅ Image redimensionnée à {scale*100:.0f}% de l’originale")
print(f"   Ancienne taille : {img.size}, Nouvelle taille : {new_size}, Résolution : {dpi}")
