import numpy as np
import matplotlib.pyplot as plt

# Rotations réelles
rotations = [15.0, 35.0, -15.5, 40.0, -20.0]
labels = [str(r) + "°" for r in rotations]

# === Résultats estimés ===
SSD  = [15.00, 34.97, -15.50, 39.92, -20.00]
NCC  = [15.00, 34.97, -15.50, 39.93, -20.00]
MI   = [15.00, 34.99, -15.50, 40.00, -20.00]
PHC  = [15.0, 35.0, -16.0, 40.0, -20.0]   # Phase corr + log-polaire
SIFT = [15.0, 35.0, -15.5, 40.0, -20.0]

methods = {
    "SSD": SSD,
    "NCC": NCC,
    "MI": MI,
    "PhaseCorr+LogPolar": PHC,
    "SIFT": SIFT
}

# === Calcul des erreurs (différence absolue) ===
def compute_rotation_errors(real, estimated):
    return [abs(r - e) for r, e in zip(real, estimated)]

errors = {m: compute_rotation_errors(rotations, est) for m, est in methods.items()}

# === Tracé comparatif ===
plt.figure(figsize=(10,6))
for m, err in errors.items():
    plt.plot(labels, err, marker='o', label=m)

plt.xlabel("Rotations appliquées (degrés)")
plt.ylabel("Erreur de localisation angulaire (degrés)")
plt.title("Comparaison des méthodes de recalage (erreur de rotation)")

# Annotation pour les courbes confondues (SSD, NCC, MI, SIFT ≈ identiques)
plt.legend()
plt.grid(True)
plt.show()
