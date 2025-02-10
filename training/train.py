# training/train.py
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import logging

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compute_loss_drgg(output, batch, model):
    """
    Calcule une loss combinée pour le modèle DRGG
    Paramètres:
      output (dict): Dictionnaire de sortie du modèle
      batch (dict): Batch de données
      model (nn.Module): Le modèle DRGG
    Retourne:
      total_loss (torch.Tensor): La loss combinée pour les objets et les relations
    """
    device = next(model.parameters()).device
    loss_obj = torch.zeros(1, device=device, requires_grad=True)
    loss_rel = torch.zeros(1, device=device, requires_grad=True)
    B = len(batch['other_data'])
    
    for i in range(B):
        ann = batch['other_data'][i]
        gt_obj = ann.get('labels_ids', [])
        gt_rel = [rel[1] for rel in ann.get('relationships_ids', [])]
        for label in gt_rel:
            if label < 0 or label >= model.num_rel_classes:
                logger.error("Label de prédicat hors limites dans l'image %d: %d (attendu entre 0 et %d)", 
                             i, label, model.num_rel_classes - 1)
                raise ValueError(f"Invalid predicate label {label} in image {i}")
        pred_obj_logits = output['obj_logits_list'][i]
        pred_rel_logits = output['rel_logits_list'][i]
        num_gt_obj = len(gt_obj)
        if num_gt_obj > 0 and pred_obj_logits.size(0) > 0:
            if pred_obj_logits.size(0) >= num_gt_obj:
                selected_obj_logits = pred_obj_logits[:num_gt_obj]
            else:
                selected_obj_logits = pred_obj_logits
                gt_obj = gt_obj[:pred_obj_logits.size(0)]
            gt_obj_tensor = torch.tensor(gt_obj, dtype=torch.long, device=device)
            if selected_obj_logits.dim() == 3 and selected_obj_logits.size(1) == 1:
                selected_obj_logits = selected_obj_logits.squeeze(1)
            loss_obj = loss_obj + F.cross_entropy(selected_obj_logits, gt_obj_tensor)
        num_gt_rel = len(gt_rel)
        if num_gt_rel > 0 and pred_rel_logits.size(0) > 0:
            if pred_rel_logits.size(0) >= num_gt_rel:
                selected_rel_logits = pred_rel_logits[:num_gt_rel]
            else:
                selected_rel_logits = pred_rel_logits
                gt_rel = gt_rel[:pred_rel_logits.size(0)]
            gt_rel_tensor = torch.tensor(gt_rel, dtype=torch.long, device=device)
            if selected_rel_logits.dim() == 3 and selected_rel_logits.size(1) == 1:
                selected_rel_logits = selected_rel_logits.squeeze(1)
            gt_rel_onehot = F.one_hot(gt_rel_tensor, num_classes=selected_rel_logits.size(-1)).float()
            loss_rel = loss_rel + F.binary_cross_entropy_with_logits(selected_rel_logits, gt_rel_onehot)
    total_loss = loss_obj + loss_rel
    return total_loss

def train_model(model, train_loader, optimizer, device, num_epochs=5, checkpoint_interval=1000):
    """
    Entraîne le modèle DRGG sur un ensemble de données et sauvegarde les meilleurs résultats
    Paramètres:
      model (nn.Module): Le modèle à entraîner
      train_loader (DataLoader): Le DataLoader d'entraînement
      optimizer (Optimizer): L'optimiseur
      device (torch.device): Le device sur lequel entraîner le modèle
      num_epochs (int, optionnel): Nombre d'époques (par défaut 5)
      checkpoint_interval (int, optionnel): Intervalle d'itérations pour sauvegarder un checkpoint (par défaut 1000)
    Retourne:
      best_loss (float): La meilleure loss moyenne obtenue sur les époques
    """
    best_loss = float('inf')
    n_batches = len(train_loader)
    all_losses = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for i, batch_data in enumerate(train_loader):
            images = [img.to(device) for img in batch_data['images']]
            output = model({
                'images': images,
                'boxes': batch_data['boxes'],
                'labels_obj': batch_data['labels_ids'],
                'rels': batch_data['relationships_ids']
            })
            loss = compute_loss_drgg(output, batch_data, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            total_loss += loss_val
            all_losses.append(loss_val)
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_path = f"checkpoint_epoch{epoch}_iter{i+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'iteration': i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                }, checkpoint_path)
                logger.info("Checkpoint sauvegardé dans %s", checkpoint_path)
            if (i + 1) % 10 == 0:
                logger.info("[Epoch %d] Step %d/%d, loss=%.4f", epoch, i+1, n_batches, loss_val)
        epoch_loss = total_loss / n_batches
        logger.info("Epoch %d - Loss moyenne : %.4f", epoch, epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_checkpoint = f"best_model_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, best_checkpoint)
            logger.info("Meilleur modèle sauvegardé dans %s", best_checkpoint)
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, label='Loss par itération')
    plt.xlabel('Itération')
    plt.ylabel('Loss')
    plt.title("Évolution de la loss pendant l'entraînement")
    plt.legend()
    plt.savefig("results/loss_plot.png")
    plt.close()
    with open("results/loss_values.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Itération", "Loss"])
        for idx, loss_val in enumerate(all_losses):
            writer.writerow([idx + 1, loss_val])
    logger.info("Plot et valeurs de loss sauvegardés dans le dossier 'results'")
    logger.info("Entraînement terminé")
    return best_loss
