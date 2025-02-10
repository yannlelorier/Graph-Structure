# utils/metrics.py
import numpy as np

def recall_at_k(pred_triplets, gt_triplets, k=50):
    """
    Calcule le Recall@K pour une image
    Paramètres:
      pred_triplets (list): Liste des triplets prédits, chaque triplet est une séquence dont le 4ème élément représente un score
      gt_triplets (list): Liste des triplets de vérité terrain
      k (int, optionnel): Le nombre de prédictions à considérer (par défaut 50)
    Retourne:
      float: Le recall calculé, c'est-à-dire le rapport entre le nombre de triplets corrects parmi les k meilleurs et le nombre total de triplets GT
    """
    if len(gt_triplets) == 0:
        return 0.0
    pred_sorted = sorted(pred_triplets, key=lambda x: x[3], reverse=True)
    top_k = pred_sorted[:k]
    correct = sum(1 for trip in top_k if tuple(trip[:3]) in {tuple(t) for t in gt_triplets})
    recall = correct / len(gt_triplets)
    return recall

def mean_recall(pred_all, gt_all, num_rel_classes=50, k=50):
    """
    Calcule la moyenne du Recall par classe (mR@K) pour un ensemble d'images
    Paramètres:
      pred_all (list): Liste de listes de triplets prédits pour chaque image
      gt_all (list): Liste de listes de triplets de vérité terrain pour chaque image
      num_rel_classes (int, optionnel): Le nombre total de classes de relations (par défaut 50)
      k (int, optionnel): Le nombre de prédictions à considérer par image (par défaut 50)
    Retourne:
      float: La moyenne des recalls par classe (mR@K)
    """
    recalls = {p: [] for p in range(num_rel_classes)}
    for pred_triplets, gt_triplets in zip(pred_all, gt_all):
        pred_sorted = sorted(pred_triplets, key=lambda x: x[3], reverse=True)[:k]
        for p in range(num_rel_classes):
            gt_p = {tuple(t) for t in gt_triplets if t[1] == p}
            if len(gt_p) == 0:
                continue
            pred_p = {tuple(t[:3]) for t in pred_sorted if t[1] == p}
            recall_p = len(gt_p & pred_p) / len(gt_p)
            recalls[p].append(recall_p)
    per_class_recall = [np.mean(recalls[p]) for p in recalls if len(recalls[p]) > 0]
    mR = np.mean(per_class_recall) if per_class_recall else 0.0
    return mR

def classification_accuracy(pred_labels, gt_labels):
    """
    Calcule la précision de classification pour les objets
    Paramètres:
      pred_labels (list): Liste des labels prédits pour les objets
      gt_labels (list): Liste des labels de vérité terrain pour les objets
    Retourne:
      float: La précision (accuracy), c'est-à-dire le ratio de prédictions correctes
    """
    if len(gt_labels) == 0:
        return 0.0
    correct = sum(1 for p, g in zip(pred_labels, gt_labels) if p == g)
    return correct / len(gt_labels)

def zero_shot_recall(pred_triplets, gt_triplets, seen_triplets, k=50):
    """
    Calcule le Zero-Shot Recall pour les relations (recall sur les triplets non vus pendant l'entraînement)
    Paramètres:
      pred_triplets (list): Liste des triplets prédits
      gt_triplets (list): Liste des triplets de vérité terrain
      seen_triplets (set): Ensemble des triplets vus lors de l'entraînement
      k (int, optionnel): Le nombre de prédictions à considérer (par défaut 50)
    Retourne:
      float: Le Zero-Shot Recall
    """
    unseen_gt = [trip for trip in gt_triplets if tuple(trip) not in seen_triplets]
    if len(unseen_gt) == 0:
        return 0.0
    pred_sorted = sorted(pred_triplets, key=lambda x: x[3], reverse=True)[:k]
    correct = sum(1 for trip in pred_sorted if tuple(trip[:3]) in {tuple(t) for t in unseen_gt})
    return correct / len(unseen_gt)

def graph_metrics(pred_triplets, gt_triplets):
    """
    Calcule les métriques au niveau du graphe (recall, précision et F1) pour les relations
    Paramètres:
      pred_triplets (list): Liste des triplets prédits
      gt_triplets (list): Liste des triplets de vérité terrain
    Retourne:
      tuple: (recall, F1) calculé sur l'ensemble des triplets
    """
    pred_set = {tuple(trip[:3]) for trip in pred_triplets}
    gt_set = {tuple(trip) for trip in gt_triplets}
    if len(gt_set) == 0:
        return 0.0, 0.0
    correct = len(pred_set & gt_set)
    recall = correct / len(gt_set)
    precision = correct / (len(pred_set) + 1e-8)
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    return recall, f1
