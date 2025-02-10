# datasets/vg_dataset.py
import os
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VGDatasetH5(Dataset):
    """
    Dataset Visual Genome utilisant un fichier HDF5
    Paramètres:
      h5_file (str): Chemin vers le fichier HDF5
      image_dir (str): Dossier contenant les images
      transform (callable, optionnel): Transformation à appliquer à l'image
      image_size (tuple, optionnel): Dimensions de redimensionnement
      obj_label2id (dict, optionnel): Mapping des labels d'objets vers des indices
      pred_label2id (dict, optionnel): Mapping des prédicats vers des indices
    """
    def __init__(self, h5_file, image_dir, transform=None, image_size=(256,256),
                 obj_label2id=None, pred_label2id=None):
        self.h5_file = h5_file
        self.image_dir = image_dir
        self.image_size = image_size
        self.obj_label2id = obj_label2id
        self.pred_label2id = pred_label2id
        self.transform = transform if transform else transforms.ToTensor()
        with h5py.File(self.h5_file, 'r') as h5f:
            self.filenames = [fn.decode('utf-8') for fn in h5f['filenames']]
            self.num_images = len(self.filenames)
            self.box_to_img = np.array(h5f['box_to_img'], dtype=np.int32)
            self.rel_to_img = np.array(h5f['rel_to_img'], dtype=np.int32)

    def __len__(self):
        """
        Retourne le nombre d'images dans le dataset
        Retourne:
          int: Nombre total d'images
        """
        return self.num_images

    def __getitem__(self, idx):
        """
        Retourne un dictionnaire avec l'image et ses annotations
        Paramètres:
          idx (int): Index de l'image à récupérer
        Retourne:
          dict: Dictionnaire contenant les clés 'image', 'boxes', 'labels_str', 'labels_ids', 'relationships_raw', 'relationships_ids' et 'image_path'
        """
        with h5py.File(self.h5_file, 'r') as h5f:
            filename = self.filenames[idx]
            box_mask = (self.box_to_img == idx)
            box_indices = np.where(box_mask)[0]
            boxes = h5f['boxes_1024'][box_indices]
            labels_bytes = h5f['labels'][box_indices]
            rel_mask = (self.rel_to_img == idx)
            rel_indices = np.where(rel_mask)[0]
            relationships_data = h5f['relationships'][rel_indices]
        labels_str = [lbl.decode('utf-8') for lbl in labels_bytes]
        if self.obj_label2id is not None:
            labels_ids = [self.obj_label2id.get(lbl, 0) for lbl in labels_str]
            labels_ids = torch.tensor(labels_ids, dtype=torch.long)
        else:
            labels_ids = None
        relationships_raw = []
        for rel in relationships_data:
            subj_id = rel['subject_id']
            predicate_str = rel['predicate'].decode('utf-8')
            obj_id = rel['object_id']
            relationships_raw.append((subj_id, predicate_str, obj_id))
        if self.pred_label2id is not None:
            relationships_ids = []
            for (s, p_str, o) in relationships_raw:
                p_id = self.pred_label2id.get(p_str, 0)
                relationships_ids.append((s, p_id, o))
        else:
            relationships_ids = None
        image_path = os.path.join(self.image_dir, os.path.basename(filename))
        pil_image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(pil_image)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        sample = {
            'image': image_tensor,
            'boxes': boxes_tensor,
            'labels_str': labels_str,
            'labels_ids': labels_ids,
            'relationships_raw': relationships_raw,
            'relationships_ids': relationships_ids,
            'image_path': image_path
        }
        return sample

def vg_collate_fn(batch):
    """
    Collate_fn personnalisé pour Visual Genome
    Paramètres:
      batch (list): Liste de dictionnaires avec les annotations de chaque image
    Retourne:
      dict: Dictionnaire regroupant les images et autres annotations pour le calcul de la loss
    """
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels_ids = [item['labels_ids'] for item in batch]
    rels_ids = [item['relationships_ids'] for item in batch]
    return {
        'images': images,
        'boxes': boxes,
        'labels_ids': labels_ids,
        'relationships_ids': rels_ids,
        'other_data': batch
    }
