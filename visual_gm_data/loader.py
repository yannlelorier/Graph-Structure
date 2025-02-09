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

    #def __iter__(self):
    #    #iter_obj = iter(self.data_obj)
    #    #import ipdb; ipdb.set_trace()
    #    #img_iter = iter_obj['image_id']
    #    return iter(self.data_obj), iter(self.data_rel)#, read_image(self.img_dir[1])

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

    
    #def __iter__(self):
    #    obj_iter = iter(self.data_obj)
    #    rel_iter = iter(self.data_rel)

    #    for obj, rel in zip(obj_iter, rel_iter):
    #        image_id = obj['image_id']

    #        # Locate image file
    #        img_path = next((path for path in self.img_paths if f'/{image_id}.jpg' in path), None)
    #        if img_path is None:
    #            continue  # Skip if image not found

    #        # Load and transform image
    #        img = read_image(img_path)
    #        if self.transform:
    #            img = self.transform(img)

    #        yield obj, rel, img  # Return a tuple (image, object annotations, relations)


class ImageTransforms:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, img):
        if self.transform:
            img = self.transform(img)
        return img

# class VisualGenomeRelationsDataset(IterableDataset):
#     def __init__(self, local=True, split='train', transform=None):
#         if local:
#             self.dataset = load_dataset('json', data_files='data_downloaded_manually/relationships.json', streaming=True)['train']
#         else:
#             self.dataset = load_dataset('visual_genome', 'relationships_v1.0.0', split=split)
#         self.transform = transform

#     def __iter__(self):
#         return iter(self.dataset)

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

    # Convert images into a batch tensor
    images = default_collate(images)  # (batch_size, C, H, W)

    # Compute max lengths for padding
    max_obj_length = max((len(obj) for obj in objects_list), default=1)
    max_rel_length = max((len(rel) for rel in relationships_list), default=1)

    # Pad objects
    # padded_objects = []
    # for obj in objects_list:
    #     obj_tensor = torch.tensor(obj + [0] * (max_obj_length - len(obj)))  # Assuming obj is a list of class indices
    #     padded_objects.append(obj_tensor)

    # padded_objects = torch.stack(padded_objects)

    # Pad relationships

    padded_images = torch.stack(padded_images)

    # padded_relationships = []
    # for rel in relationships_list:
    #     rel_tensor = torch.tensor(rel + [0] * (max_rel_length - len(rel)))  # Assuming rel is a list of relation indices
    #     padded_relationships.append(rel_tensor)

    # padded_relationships = torch.stack(padded_relationships)

    max_obj_length = max((len(obj) for obj in objects_list), default=1)
    max_rel_length = max((len(rel) for rel in relationships_list), default=1)

    padded_objects = [torch.tensor(obj + [0] * (max_obj_length - len(obj))) for obj in objects_list]
    padded_relationships = [torch.tensor(rel + [0] * (max_rel_length - len(rel))) for rel in relationships_list]


    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    padded_images = []
    for img in images:
        c, h, w = img.shape
        padded_img = torch.zeros((c, max_h, max_w))  # Create zero-padded image
        padded_img[:, :h, :w] = img  # Copy original image
        padded_images.append(padded_img)

    return {
        "images": padded_images,  # Padded images (batch_size, 3, max_h, max_w)
        "image_ids": image_ids,
        "objects": torch.stack(padded_objects),
        "relationships": torch.stack(padded_relationships),
    }
    # return {
    #     "images": images,  # Tensor of images (batch_size, C, H, W)
    #     "image_ids": image_ids,  # List of image IDs
    #     "objects": padded_objects,  # Padded tensor (batch_size, max_obj_length)
    #     "relationships": padded_relationships,  # Padded tensor (batch_size, max_rel_length)
    # }

def pad_collate_fn_2(batch):
    """
    Custom collate function for Visual Genome.
    1. Pads variable-length `relationships` and `objects`.
    2. Converts `image_id` to a list (not tensor).
    3. Uses `default_collate` for images.
    """

    images, objects_list, relationships_list, image_ids = zip(*batch)

    # Extract image tensors (let PyTorch handle them)
    images = default_collate(images)

    # Compute max lengths
    max_obj_length = max(len(obj) for obj in objects_list)
    max_rel_length = max(len(rel) for rel in relationships_list)

    # Pad objects
    padded_objects = []
    for obj in objects_list:
        obj_tensor = torch.tensor(obj + [0] * (max_obj_length - len(obj)))  # Assuming obj is a list of class indices
        padded_objects.append(obj_tensor)

    padded_objects = torch.stack(padded_objects)

    # Pad relationships
    padded_relationships = []
    for rel in relationships_list:
        rel_tensor = torch.tensor(rel + [0] * (max_rel_length - len(rel)))  # Assuming rel is a list of relation indices
        padded_relationships.append(rel_tensor)

    padded_relationships = torch.stack(padded_relationships)

    return {
        "images": images,  # Tensor of images (batch_size, C, H, W)
        "image_ids": list(image_ids),  # List of image IDs (not tensor)
        "objects": padded_objects,  # Padded tensor (batch_size, max_obj_length)
        "relationships": padded_relationships,  # Padded tensor (batch_size, max_rel_length)
    }

def pad_collate_fn(batch):
    """
    Custom collate function that:
    1. Extracts `image_id`
    2. Converts `relationships` into tensors
    3. Pads sequences to max length in the batch
    """

    image_ids = [item["image_id"] for item in batch]

    relationships_list = [item["relationships"] for item in batch]
    objects_list = [item["objects"] for item in batch]

    # Convert relationships to tensor representations (Example: using their lengths)
    relationship_lengths = torch.tensor([len(rel) for rel in relationships_list])  # Just an example
    object_lengths = torch.tensor([len(obj) for obj in objects_list])  # Just an example

    # If each relationship is a dictionary, we need to extract meaningful features
    max_rel_length = max(relationship_lengths).item()  # Find max length in batch
    padded_relationships = []

    max_obj_length = max(object_lengths).item()  # Find max length in batch
    padded_objects = []

    for obj in objects_list:
        # Example: Convert each relation to a fixed-size vector (e.g., one-hot or embeddings)
        # Here, we just store the count of relations in a tensor
        # Note: You can use a more sophisticated method to encode relationships
        obj_tensor = torch.tensor([len(obj)] + [0] * (max_obj_length - len(obj)))  # Pad with 0s
        padded_objects.append(obj_tensor)

    padded_objects = pad_sequence(padded_objects, batch_first=True, padding_value=0)

    for rel in relationships_list:
        # Example: Convert each relation to a fixed-size vector (e.g., one-hot or embeddings)
        # Here, we just store the count of relations in a tensor
        # Note: You can use a more sophisticated method to encode relationships
        rel_tensor = torch.tensor([len(rel)] + [0] * (max_rel_length - len(rel)))  # Pad with 0s
        padded_relationships.append(rel_tensor)

    padded_relationships = pad_sequence(padded_relationships, batch_first=True, padding_value=0)


    return {
        "image_id": torch.tensor(image_ids),
        "relationships": padded_relationships,  # Padded tensor
        "objects": padded_objects,  # Padded tensor
    }

def load_visual_genome_data(batch_size=4, shuffle=True, split='train', num_workers=4, transform=None):
    dataset = VisualGenomeDataset(split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_collate_fn)
    return dataloader

# Test with smaller datasets
def test_loader_with_small_dataset():
    dataset = VisualGenomeDataset(split='train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=pad_collate_fn_3)
    
    return dataloader

if __name__ == "__main__":
    test_loader_with_small_dataset()
