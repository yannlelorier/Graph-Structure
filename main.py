# main.py
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.vg_dataset import VGDatasetH5, vg_collate_fn
from models.sgg_model import MyRelModelDRGG
from training.train import train_model

def main():
    # Charger le fichier de configuration (ici, on charge aussi les mappings à partir du JSON)
    with open("data/VG-SGG-dicts.json", "r") as f:
        json_data = json.load(f)

    # Création des mappings
    obj_label2id = {label: idx for label, idx in json_data["label_to_idx"].items()}
    pred_label2id = {predicate: (idx - 1) for predicate, idx in json_data["predicate_to_idx"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")

    train_dataset = VGDatasetH5(
        h5_file="data/VG-SGG.h5",
        image_dir="data/VG_100K",
        transform=None,
        image_size=(256, 256),
        obj_label2id=obj_label2id,
        pred_label2id=pred_label2id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=vg_collate_fn,
        num_workers=4
    )

    model = MyRelModelDRGG(
        num_obj_classes=151,
        num_rel_classes=50,
        hidden_dim=256,
        mode='sgdet',
        use_bias=True,
        test_bias=False,
        num_layers=3,
        roi_size=7
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Entraînement du modèle sur 5 époques avec sauvegarde périodique et logging
    best_loss = train_model(model, train_loader, optimizer, device, num_epochs=5, checkpoint_interval=1000)
    print(f"Meilleure loss obtenue: {best_loss:.4f}")

if __name__ == "__main__":
    main()
