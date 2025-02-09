import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import read_image
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataloader import default_collate
from datasets import load_dataset
import glob


class VisualGenomeDataset(IterableDataset):
    def __init__(self, local=True, split='train', transform=None, img_dirs=['data_downloaded_manually/VG_100K', 'data_downloaded_manually/VG_100K_2'], img_size=(512, 512)):
        if local:
            self.data_obj = load_dataset('json', data_files='data_downloaded_manually/objects.json', streaming=True)['train']
            self.data_rel = load_dataset('json', data_files='data_downloaded_manually/relationships.json', streaming=True)['train']
            self.img_files = sorted(glob.glob(img_dirs[0]+'/*.jpg')) + sorted(glob.glob(img_dirs[1]+'/*.jpg'))
            self.image_size = img_size
        else:
            #TODO
            raise NotImplementedError("Remote data loading not implemented yet")
            #self.dataset = load_dataset('visual_genome', 'objects_v1.0.0', split=split)
        self.transform = transform

    def __iter__(self):
        obj_iter = iter(self.data_obj)
        rel_iter = iter(self.data_rel)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts the PIL image to a tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes the tensor
        ])

        for obj_item, rel_item in zip(obj_iter, rel_iter):
            image_id = obj_item["image_id"]
            img_path = next((img for img in self.img_files if str(image_id) in img), None)

            if img_path:
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                #image = read_image(img_path).float() / 255.0  # Normalize image
                if self.transform:
                    image = self.transform(image)

                yield {
                    "image_id": image_id,
                    "objects": obj_item["objects"],  # Assuming list of object annotations
                    "relationships": rel_item["relationships"],  # Assuming list of relationship annotations
                    "image": image,
                }

    
class ImageTransforms:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, img):
        if self.transform:
            img = self.transform(img)
        return img


def pad_collate_fn_3(batch):
    """
    Custom collate function for Visual Genome.
    1. Extracts `image_id`, `objects`, and `relationships`.
    2. Pads variable-length objects and relationships.
    3. Uses `default_collate` for images.
    """

    images = [item["image"] for item in batch]
    image_ids = [item["image_id"] for item in batch]  # Now correctly extracted
    objects_list = [item.get("objects", []) for item in batch]  # Handle missing objects
    relationships_list = [item.get("relationships", []) for item in batch]  # Handle missing rels

    # Compute max lengths for padding
    max_obj_length = max((len(obj) for obj in objects_list), default=1)
    max_rel_length = max((len(rel) for rel in relationships_list), default=1)

    #padded_objects = [obj + [0] * (max_obj_length - len(obj)) for obj in objects_list]
    #padded_relationships = [rel + [0] * (max_rel_length - len(rel)) for rel in relationships_list]

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        c, h, w = img.shape
        padded_img = torch.zeros((c, max_h, max_w))  # Create zero-padded image
        padded_img[:, :h, :w] = img  # Copy original image
        padded_images.append(padded_img)

    padded_objects = []
    padded_relationships = []
    for obj_list, rel_list in zip(objects_list, relationships_list):
        # Pad objects
        padded_obj = []
        for obj in obj_list:
            padded_obj.append([
                obj.get('x', 0), obj.get('y', 0), obj.get('w', 0), obj.get('h', 0),
                obj.get('name', ''), obj.get('synsets', '')
            ])
        while len(padded_obj) < max_obj_length:
            padded_obj.append([0, 0, 0, 0, '', ''])
        padded_objects.append(padded_obj)

        # Pad relationships
        padded_rel = []
        for rel in rel_list:
            padded_rel.append([
                rel.get('subject_id', 0), rel.get('object_id', 0), rel.get('predicate', '')
            ])
        while len(padded_rel) < max_rel_length:
            padded_rel.append([0, 0, ''])
        padded_relationships.append(padded_rel)
    return {
        "images": torch.stack(tuple(padded_images)),  # Padded images (batch_size, 3, max_h, max_w)
        "image_ids": image_ids,
        "objects": padded_objects,
        "relationships": padded_relationships,
    }

def load_visual_genome_data(batch_size=4, shuffle=True, split='train', num_workers=4, transform=None):
    dataset = VisualGenomeDataset(split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_collate_fn_3)
    return dataloader

# Test with smaller datasets
def test_loader_with_small_dataset():
    dataset = VisualGenomeDataset(split='train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=pad_collate_fn_3)
    
    return dataloader

if __name__ == "__main__":
    test_loader_with_small_dataset()
