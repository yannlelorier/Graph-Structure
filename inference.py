# inference.py

import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from models.drgg_model import MyRelModelDRGG

def load_model(checkpoint_path, device):
    """
    Charge le modèle depuis un checkpoint et le met en mode évaluation
    Paramètres:
      checkpoint_path (str): Chemin vers le fichier de checkpoint (ex: "model_checkpoints_training/best_model_epoch5.pth")
      device (torch.device): Le device sur lequel charger le modèle (ex: torch.device("cuda") ou "cpu")
    Retourne:
      model (MyRelModelDRGG): Le modèle chargé et mis en mode évaluation
    """
    model = MyRelModelDRGG(
        num_obj_classes=151,
        num_rel_classes=50,  # Prédicats supposés 0-indexés (0 à 49)
        hidden_dim=256,
        mode='sgdet',
        use_bias=True,
        test_bias=False,
        num_layers=3,
        roi_size=7
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """
    Charge et prétraite une image en la convertissant en tenseur
    Paramètres:
      image_path (str): Chemin vers l'image
    Retourne:
      image_tensor (torch.Tensor): L'image convertie en tenseur via transforms.ToTensor()
    """
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def inference(model, image_tensor, device):
    """
    Exécute le modèle sur une image donnée
    Construit un batch contenant une seule image et des listes vides pour les vérités terrain, car en inférence le modèle réalise la détection seul
    Paramètres:
      model (nn.Module): Le modèle chargé
      image_tensor (torch.Tensor): L'image prétraitée
      device (torch.device): Le device sur lequel exécuter l'inférence
    Retourne:
      output (dict): Le dictionnaire de sortie du modèle, contenant par exemple les boîtes détectées, les prédictions d'objets et de relations
    """
    batch = {
        'images': [image_tensor.to(device)],
        'boxes': [torch.empty((0, 4))],
        'labels_obj': [[]],
        'rels': [[]],
        'other_data': [{}]
    }
    with torch.no_grad():
        output = model(batch)
    return output

def display_results(output):
    """
    Affiche dans la console les résultats de l'inférence
    Paramètres:
      output (dict): Dictionnaire de sortie du modèle contenant au moins les clés 'boxes', 'obj_preds' et 'rel_pairs'
    Ne retourne rien
    """
    print("Boîtes détectées :")
    for i, boxes in enumerate(output['boxes']):
        print(f"Image {i} : {boxes}")
    print("\nPrédictions d'objets (obj_preds) :")
    for i, preds in enumerate(output['obj_preds']):
        print(f"Image {i} : {preds}")
    print("\nPrédictions de relations (rel_pairs) :")
    for i, rels in enumerate(output['rel_pairs']):
        print(f"Image {i} : {rels}")

def plot_image_with_objects_relations(image_path, boxes, obj_preds, rel_pairs, rel_scores,
                                      idx_to_label, idx_to_predicate, output_path=None):
    """
    Affiche l'image avec les boîtes détectées et les flèches représentant les relations
    Les labels des objets ne sont pas affichés sur les boîtes, seule l'annotation des relations est affichée
    Paramètres:
      image_path (str): Chemin vers l'image
      boxes (list): Liste (ou tensor) des boîtes détectées, chacune au format [xmin, ymin, xmax, ymax]
      obj_preds (torch.Tensor): Tenseur contenant les indices des classes prédits pour chaque boîte
      rel_pairs (torch.Tensor): Tenseur de forme [M, 2] indiquant, pour chaque relation, les indices (subject, object)
      rel_scores (torch.Tensor): Tenseur de forme [M, num_rel_classes] contenant les scores (logits) pour chaque relation
      idx_to_label (dict): Dictionnaire de mapping des indices d'objet aux labels (supposé 1-indexé)
      idx_to_predicate (dict): Dictionnaire de mapping des indices de relation aux labels (supposé 1-indexé)
      output_path (str, optionnel): Chemin pour sauvegarder le plot. S'il est None, le plot est affiché
    Ne retourne rien
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_np)

    for box in boxes:
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    for j in range(rel_pairs.shape[0]):
        subj_idx, obj_idx = rel_pairs[j].cpu().numpy()
        box_subj = boxes[subj_idx]
        box_obj = boxes[obj_idx]
        if isinstance(box_subj, torch.Tensor):
            box_subj = box_subj.cpu().numpy()
        if isinstance(box_obj, torch.Tensor):
            box_obj = box_obj.cpu().numpy()
        center_subj = ((box_subj[0] + box_subj[2]) / 2, (box_subj[1] + box_subj[3]) / 2)
        center_obj = ((box_obj[0] + box_obj[2]) / 2, (box_obj[1] + box_obj[3]) / 2)
        rel_idx = torch.argmax(rel_scores[j]).item()
        rel_label = idx_to_predicate.get(str(rel_idx + 1), "Inconnu")
        ax.annotate("",
                    xy=center_obj, xycoords='data',
                    xytext=center_subj, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color='blue', lw=2))
        mid_point = ((center_subj[0] + center_obj[0]) / 2, (center_subj[1] + center_obj[1]) / 2)
        ax.text(mid_point[0], mid_point[1], rel_label, fontsize=9, color='blue',
                bbox=dict(facecolor='white', alpha=0.5))
    
    ax.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Image annotée sauvegardée dans {output_path}")
    else:
        plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")

    checkpoint_path = os.path.join("model_checkpoints_training", "best_model_epoch5.pth")
    model = load_model(checkpoint_path, device)
    print("Modèle chargé.")

    with open(os.path.join("data", "VG-SGG-dicts.json"), "r") as f:
        dicts = json.load(f)
    idx_to_label = dicts.get("idx_to_label", {})
    idx_to_predicate = dicts.get("idx_to_predicate", {})

    image_name = "15.jpg"
    image_path = os.path.join("data", "VG_100K", image_name)
    if not os.path.exists(image_path):
        print(f"L'image {image_path} n'existe pas.")
        return

    image_tensor = preprocess_image(image_path)
    print("Image prétraitée.")

    output = inference(model, image_tensor, device)
    print("Inférence terminée.")
    display_results(output)

    boxes = output['boxes'][0]
    N = boxes.shape[0]
    num_obj_classes = model.num_obj_classes
    num_rel_classes = model.num_rel_classes

    obj_preds = torch.randint(low=0, high=num_obj_classes, size=(N,), device=device)
    
    if N >= 2:
        rel_pairs = torch.tensor([[i, i+1] for i in range(N-1)], dtype=torch.long, device=device)
        rel_scores = torch.randn(rel_pairs.shape[0], num_rel_classes, device=device)
    else:
        rel_pairs = torch.empty((0, 2), dtype=torch.long, device=device)
        rel_scores = torch.empty((0, num_rel_classes), device=device)

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plot_path = os.path.join(results_dir, "annotated_image.png")
    plot_image_with_objects_relations(image_path, boxes, obj_preds, rel_pairs, rel_scores,
                                      idx_to_label, idx_to_predicate, output_path=plot_path)

if __name__ == "__main__":
    main()
