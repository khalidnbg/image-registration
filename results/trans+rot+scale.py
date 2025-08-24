import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Configuration du style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paramètres réels
params_real = {
    "tx": 15.55, 
    "ty": 20.60, 
    "angle": 35.0, 
    "scale": 1.25
}

# Méthodes et estimations avec noms complets
methods = {
    "SSD": {"tx": 17.55, "ty": 0.94, "angle": 5.31, "scale": 0.86, "name": "Sum of Squared Differences"},
    "NCC": {"tx": 0.68, "ty": 20.73, "angle": 34.98, "scale": 0.80, "name": "Normalized Cross Correlation"},
    "MI":  {"tx": 10.00, "ty": 15.00, "angle": 30.00, "scale": 1.20, "name": "Mutual Information"},
    "PhaseCorr": {"tx": -158.95, "ty": 642.05, "angle": 34.9922, "scale": 1.2356, "name": "Phase Correlation"},
    "SIFT+RANSAC": {"tx": 15.7, "ty": 20.4, "angle": 35.00, "scale": 1.2499, "name": "SIFT + RANSAC"}
}

# Couleurs cohérentes pour chaque méthode
colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
method_colors = {method: colors[i] for i, method in enumerate(methods.keys())}

# Fonction pour calculer les erreurs
def calculate_errors():
    errors = {}
    for method_name, values in methods.items():
        errors[method_name] = {}
        for param in ["tx", "ty", "angle", "scale"]:
            real_val = params_real[param]
            est_val = values[param]
            errors[method_name][param] = abs(est_val - real_val)
    return errors

errors = calculate_errors()

# =============================================================================
# 1. DIAGRAMME: TRANSLATION X
# =============================================================================
def plot_translation_x():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    param = "tx"
    real_value = params_real[param]
    method_names = list(methods.keys())
    values = [methods[m][param] for m in method_names]
    
    # Barres avec couleurs personnalisées
    bars = ax.bar(range(len(method_names)), values, 
                  color=[method_colors[m] for m in method_names],
                  alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Ligne de référence
    ax.axhline(real_value, color='red', linestyle='--', linewidth=3, 
               label=f'Valeur réelle: {real_value} px', zorder=10)
    
    # Zone de tolérance (±5%)
    tolerance = abs(real_value * 0.05)
    ax.axhspan(real_value - tolerance, real_value + tolerance, 
               alpha=0.15, color='green', label=f'Zone de tolérance (±5%): [{real_value-tolerance:.1f}, {real_value+tolerance:.1f}]')
    
    # Annotations des valeurs et erreurs
    for i, (bar, value, method) in enumerate(zip(bars, values, method_names)):
        height = bar.get_height()
        error = errors[method][param]
        
        # Valeur sur la barre
        ax.annotate(f'{value:.2f} px', 
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 10 if height >= 0 else -25),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=11, weight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Erreur sous la barre
        ax.text(i, min(values) - (max(values) - min(values)) * 0.15,
                f'Erreur: {error:.2f} px\n({(error/real_value)*100:.1f}%)',
                ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=method_colors[method], alpha=0.3))
    
    # Mise en forme
    ax.set_title('Translation X - Comparaison des Méthodes de Recalage', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Translation X (pixels)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Méthodes de Recalage', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([methods[m]["name"] for m in method_names], 
                       rotation=15, ha='right', fontsize=10)
    
    # Légende améliorée
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Ajustement des limites
    y_range = max(values) - min(values)
    ax.set_ylim(min(values) - y_range * 0.3, max(values) + y_range * 0.2)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. DIAGRAMME: TRANSLATION Y
# =============================================================================
def plot_translation_y():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    param = "ty"
    real_value = params_real[param]
    method_names = list(methods.keys())
    values = [methods[m][param] for m in method_names]
    
    # Barres avec couleurs personnalisées
    bars = ax.bar(range(len(method_names)), values, 
                  color=[method_colors[m] for m in method_names],
                  alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Ligne de référence
    ax.axhline(real_value, color='red', linestyle='--', linewidth=3, 
               label=f'Valeur réelle: {real_value} px', zorder=10)
    
    # Zone de tolérance
    tolerance = abs(real_value * 0.05)
    ax.axhspan(real_value - tolerance, real_value + tolerance, 
               alpha=0.15, color='green', label=f'Zone de tolérance (±5%)')
    
    # Annotations
    for i, (bar, value, method) in enumerate(zip(bars, values, method_names)):
        height = bar.get_height()
        error = errors[method][param]
        
        ax.annotate(f'{value:.2f} px', 
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 10 if height >= 0 else -25),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=11, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.text(i, min(values) - (max(values) - min(values)) * 0.15,
                f'Erreur: {error:.2f} px\n({(error/real_value)*100:.1f}%)',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=method_colors[method], alpha=0.3))
    
    ax.set_title('Translation Y - Comparaison des Méthodes de Recalage', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Translation Y (pixels)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Méthodes de Recalage', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([methods[m]["name"] for m in method_names], 
                       rotation=15, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    y_range = max(values) - min(values)
    ax.set_ylim(min(values) - y_range * 0.3, max(values) + y_range * 0.2)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. DIAGRAMME: ROTATION (ANGLE)
# =============================================================================
def plot_rotation():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    param = "angle"
    real_value = params_real[param]
    method_names = list(methods.keys())
    values = [methods[m][param] for m in method_names]
    
    bars = ax.bar(range(len(method_names)), values, 
                  color=[method_colors[m] for m in method_names],
                  alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.axhline(real_value, color='red', linestyle='--', linewidth=3, 
               label=f'Valeur réelle: {real_value}°', zorder=10)
    
    tolerance = abs(real_value * 0.05)
    ax.axhspan(real_value - tolerance, real_value + tolerance, 
               alpha=0.15, color='green', label=f'Zone de tolérance (±5%)')
    
    for i, (bar, value, method) in enumerate(zip(bars, values, method_names)):
        height = bar.get_height()
        error = errors[method][param]
        
        ax.annotate(f'{value:.2f}°', 
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 10 if height >= 0 else -25),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=11, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.text(i, min(values) - (max(values) - min(values)) * 0.15,
                f'Erreur: {error:.2f}°\n({(error/real_value)*100:.1f}%)',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=method_colors[method], alpha=0.3))
    
    ax.set_title('Rotation (Angle) - Comparaison des Méthodes de Recalage', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Angle de rotation (degrés)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Méthodes de Recalage', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([methods[m]["name"] for m in method_names], 
                       rotation=15, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    y_range = max(values) - min(values)
    ax.set_ylim(min(values) - y_range * 0.3, max(values) + y_range * 0.2)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. DIAGRAMME: ÉCHELLE
# =============================================================================
def plot_scale():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    param = "scale"
    real_value = params_real[param]
    method_names = list(methods.keys())
    values = [methods[m][param] for m in method_names]
    
    bars = ax.bar(range(len(method_names)), values, 
                  color=[method_colors[m] for m in method_names],
                  alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.axhline(real_value, color='red', linestyle='--', linewidth=3, 
               label=f'Valeur réelle: {real_value}', zorder=10)
    
    tolerance = abs(real_value * 0.05)
    ax.axhspan(real_value - tolerance, real_value + tolerance, 
               alpha=0.15, color='green', label=f'Zone de tolérance (±5%)')
    
    for i, (bar, value, method) in enumerate(zip(bars, values, method_names)):
        height = bar.get_height()
        error = errors[method][param]
        
        ax.annotate(f'{value:.4f}', 
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 10 if height >= 0 else -25),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=11, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.text(i, min(values) - (max(values) - min(values)) * 0.15,
                f'Erreur: {error:.4f}\n({(error/real_value)*100:.1f}%)',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=method_colors[method], alpha=0.3))
    
    ax.set_title('Facteur d\'Échelle - Comparaison des Méthodes de Recalage', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Facteur d\'échelle', fontsize=12, fontweight='bold')
    ax.set_xlabel('Méthodes de Recalage', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([methods[m]["name"] for m in method_names], 
                       rotation=15, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    y_range = max(values) - min(values)
    ax.set_ylim(min(values) - y_range * 0.3, max(values) + y_range * 0.2)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 5. DIAGRAMME: HEATMAP DES ERREURS
# =============================================================================
def plot_error_heatmap():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Matrice des erreurs
    error_matrix = np.array([[errors[method][param] for param in ["tx", "ty", "angle", "scale"]] 
                            for method in methods.keys()])
    
    # Heatmap avec annotations
    im = ax.imshow(error_matrix, cmap='Reds', aspect='auto', interpolation='nearest')
    
    # Annotations détaillées
    for i in range(len(methods)):
        for j, param in enumerate(["tx", "ty", "angle", "scale"]):
            error_val = error_matrix[i, j]
            real_val = params_real[param]
            rel_error = (error_val / real_val) * 100
            
            text = ax.text(j, i, f'{error_val:.2f}\n({rel_error:.1f}%)',
                          ha="center", va="center", color="black" if error_val < np.max(error_matrix)/2 else "white", 
                          fontweight='bold', fontsize=10)
    
    # Configuration des axes
    ax.set_title('Matrice des Erreurs Absolues par Méthode et Paramètre', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Translation X\n(pixels)', 'Translation Y\n(pixels)', 
                       'Rotation\n(degrés)', 'Facteur\nd\'Échelle'], fontsize=11)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([methods[m]["name"] for m in methods.keys()], fontsize=11)
    
    # Barre de couleur
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Erreur absolue', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Bordures pour séparer les cellules
    for i in range(len(methods) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
    for j in range(5):
        ax.axvline(j - 0.5, color='white', linewidth=2)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. DIAGRAMME: CLASSEMENT ET SCORES GLOBAUX
# =============================================================================
def plot_ranking():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calcul du score global (erreur moyenne normalisée)
    global_scores = {}
    for method in methods.keys():
        total_error = 0
        for param in ["tx", "ty", "angle", "scale"]:
            normalized_error = errors[method][param] / params_real[param]
            total_error += normalized_error
        global_scores[method] = total_error / 4
    
    # Classement
    ranked_methods = sorted(global_scores.items(), key=lambda x: x[1])
    
    # Graphique 1: Scores globaux
    method_names = [item[0] for item in ranked_methods]
    scores = [item[1] for item in ranked_methods]
    
    bars1 = ax1.barh(range(len(method_names)), scores, 
                     color=[method_colors[m] for m in method_names],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Annotations des scores
    for i, (bar, score, method) in enumerate(zip(bars1, scores, method_names)):
        width = bar.get_width()
        ax1.annotate(f'{score:.4f}', 
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=11, weight='bold')
        
        # Rang
        ax1.text(-max(scores)*0.05, i, f'#{i+1}', 
                ha='right', va='center', fontsize=12, weight='bold',
                bbox=dict(boxstyle="circle,pad=0.3", facecolor='gold' if i==0 else 'lightgray'))
    
    ax1.set_title('Classement des Méthodes\n(Score d\'Erreur Moyenne Normalisée)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Score d\'Erreur (plus bas = meilleur)', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(len(method_names)))
    ax1.set_yticklabels([methods[m]["name"] for m in method_names], fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, max(scores) * 1.2)
    
    # Graphique 2: Répartition des erreurs par paramètre
    param_labels = ['Translation X', 'Translation Y', 'Rotation', 'Échelle']
    best_method = ranked_methods[0][0]
    best_errors = [errors[best_method][param] for param in ["tx", "ty", "angle", "scale"]]
    
    # Normalisation pour comparaison
    normalized_errors = [errors[best_method][param] / params_real[param] * 100 
                        for param in ["tx", "ty", "angle", "scale"]]
    
    bars2 = ax2.bar(param_labels, normalized_errors, 
                   color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, error in zip(bars2, normalized_errors):
        height = bar.get_height()
        ax2.annotate(f'{error:.1f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, weight='bold')
    
    ax2.set_title(f'Distribution des Erreurs Relatives\nMeilleure Méthode: {methods[best_method]["name"]}', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('Erreur Relative (%)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(param_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# AFFICHAGE DE TOUS LES DIAGRAMMES
# =============================================================================

print("="*80)
print("ANALYSE COMPARATIVE DES MÉTHODES DE RECALAGE D'IMAGES")
print("="*80)
print(f"Paramètres de référence:")
print(f"• Translation X: {params_real['tx']} pixels")
print(f"• Translation Y: {params_real['ty']} pixels") 
print(f"• Rotation: {params_real['angle']}°")
print(f"• Facteur d'échelle: {params_real['scale']}")
print("="*80)

print("\n1. Affichage du diagramme Translation X...")
plot_translation_x()

print("\n2. Affichage du diagramme Translation Y...")
plot_translation_y()

print("\n3. Affichage du diagramme Rotation...")
plot_rotation()

print("\n4. Affichage du diagramme Facteur d'Échelle...")
plot_scale()

print("\n5. Affichage de la Heatmap des Erreurs...")
plot_error_heatmap()

print("\n6. Affichage du Classement Global...")
plot_ranking()

print("\n" + "="*80)
print("ANALYSE TERMINÉE - Tous les diagrammes ont été affichés individuellement")
print("="*80)