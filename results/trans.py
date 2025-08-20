import numpy as np
import matplotlib.pyplot as plt

# Translations appliquées
translations = [(10,5), (-15,8), (20,-10), (-25,-15), (30,12)]
labels = [f"({tx},{ty})" for tx,ty in translations]

# === Résultats estimés (remplis à partir de tes logs) ===
SSD = [(10.00,5.00), (-14.99,8.01), (20.01,-9.99), (-25.02,-14.99), (30.01,12.01)]
NCC = [(10.01,5.00), (-14.99,7.99), (20.00,-9.99), (-24.99,-15.01), (30.01,12.01)]
MI  = [(9.99,5.01), (-15.00,8.00), (20.01,-10.01), (-24.99,-15.00), (29.99,12.01)]
COR_PHC_SIFT = [(10,5), (-15,8), (20,-10), (-25,-15), (30,12)]
# PHC = [(10,5), (-15,8), (20,-10), (-25,-15), (30,12)]
# SIFT= [(10,5), (-15,8), (20,-10), (-25,-15), (30,12)]

methods = {
    "SSD": SSD,
    "NCC": NCC,
    "MI": MI,
    "Correlation, PhaseCorr, SIFT+RANSAC": COR_PHC_SIFT,
}

# === Calcul des erreurs ===
def compute_errors(translations, estimated):
    errors = []
    for (tx,ty),(ex,ey) in zip(translations,estimated):
        e = np.sqrt((tx-ex)**2 + (ty-ey)**2)
        errors.append(e)
    return errors

errors = {m: compute_errors(translations, est) for m,est in methods.items()}

# === Tracé comparatif ===
plt.figure(figsize=(10,6))
for m, err in errors.items():
    plt.plot(labels, err, marker='o', label=m)

plt.xlabel("Translations appliquées (Tx,Ty)")
plt.ylabel("Erreur de localisation (pixels)")
plt.title("Comparaison des méthodes de recalage (erreur de translation)")
plt.legend()
plt.grid(True)
plt.show()
