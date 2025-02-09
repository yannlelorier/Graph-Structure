import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import glob
import ijson
import random
import logging
from logger.my_logger import CustomFormatter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


class VisualGenomeDataset(IterableDataset):
    def __init__(self, local=True, img_dirs=['data_downloaded_manually/VG_100K', 'data_downloaded_manually/VG_100K_2'], img_size=(512, 512), shuffle=True):
        if local:
            logger.info("Loading local data...")
            self.obj_file = 'data_downloaded_manually/objects.json'
            self.rel_file = 'data_downloaded_manually/relationships.json'
            self.img_files = sorted(glob.glob(img_dirs[0] + '/*.jpg')) + sorted(glob.glob(img_dirs[1] + '/*.jpg'))
            self.image_size = img_size
            self.shuffle = shuffle
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise NotImplementedError("Remote data loading not implemented yet")

    def parse_json_stream(self, file_path):
        """Streams large JSON files where data is stored as a list (not JSONL)."""
        with open(file_path, 'r') as file:
            for obj in ijson.items(file, "item"):
                yield obj

    def __iter__(self):
        obj_iter = self.parse_json_stream(self.obj_file)
        rel_iter = self.parse_json_stream(self.rel_file)
        
        # If shuffling is enabled, load everything into memory first (limited by dataset size)
        if self.shuffle:
            obj_list = list(obj_iter)
            rel_list = list(rel_iter)
            combined = list(zip(obj_list, rel_list))
            random.shuffle(combined)
            obj_iter, rel_iter = zip(*combined)
            obj_iter = iter(obj_iter)
            rel_iter = iter(rel_iter)

        for obj_item, rel_item in zip(obj_iter, rel_iter):
            logger.info(f"obj_item: {len(obj_item)}")
            image_id = obj_item.get("image_id")
            img_path = next((img for img in self.img_files if str(image_id) in img), None)

            if img_path:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = self.transform(image)
                except Exception as e:
                    logger.warning(f"Error loading image {img_path}: {e}")
                    continue  # Skip this entry if image loading fails

                yield {
                    "image_id": image_id,
                    "objects": obj_item.get("objects", []),
                    "relationships": rel_item.get("relationships", []),
                    "image": image,
                }
            else:
                logger.warning(f"Image file for image_id {image_id} not found.")


def pad_collate_fn(batch):
    """
    Custom collate function to pad images, objects, and relationships.
    """
    images = [item["image"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    objects_list = [item.get("objects", []) for item in batch]
    relationships_list = [item.get("relationships", []) for item in batch]

    # Find max dimensions
    max_obj_length = max((len(obj) for obj in objects_list), default=1)
    max_rel_length = max((len(rel) for rel in relationships_list), default=1)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    # Pad images to (C, max_h, max_w)
    padded_images = torch.zeros((len(images), 3, max_h, max_w))
    for i, img in enumerate(images):
        c, h, w = img.shape
        padded_images[i, :, :h, :w] = img

    # Pad objects and relationships
    padded_objects = torch.zeros((len(objects_list), max_obj_length, 4))  # Only bounding box info
    padded_relationships = torch.zeros((len(relationships_list), max_rel_length, 2))  # Subject/Object IDs

    for i, obj_list in enumerate(objects_list):
        for j, obj in enumerate(obj_list[:max_obj_length]):  # Truncate if too long
            padded_objects[i, j, :] = torch.tensor([
                obj.get('x', 0), obj.get('y', 0), obj.get('w', 0), obj.get('h', 0)
            ])

    for i, rel_list in enumerate(relationships_list):
        for j, rel in enumerate(rel_list[:max_rel_length]):
            padded_relationships[i, j, :] = torch.tensor([
                rel.get('subject_id', 0), rel.get('object_id', 0)
            ])

    return {
        "images": padded_images,  # Shape: (batch_size, 3, H, W)
        "image_ids": torch.tensor(image_ids),
        "objects": padded_objects,  # Shape: (batch_size, max_obj_length, 4)
        "relationships": padded_relationships,  # Shape: (batch_size, max_rel_length, 2)
    }


def load_visual_genome_data(batch_size=4, num_workers=4):
    dataset = VisualGenomeDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=pad_collate_fn)
    return dataloader


def test_loader_with_small_dataset():
    dataloader = load_visual_genome_data(batch_size=1, num_workers=1)
    return dataloader


if __name__ == "__main__":
    test_loader_with_small_dataset()
