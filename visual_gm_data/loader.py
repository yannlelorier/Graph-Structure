import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset


class VisualGenomeObjectsDataset(IterableDataset):
    def __init__(self, local=True, split='train', transform=None):
        if local:
            self.dataset = load_dataset('json', data_files='data_downloaded_manually/objects.json', streaming=True)['train']
        else:
            self.dataset = load_dataset('visual_genome', 'objects_v1.0.0', split=split)
        self.transform = transform

    def __iter__(self):
        return iter(self.dataset)


class VisualGenomeRelationsDataset(IterableDataset):
    def __init__(self, local=True, split='train', transform=None):
        if local:
            self.dataset = load_dataset('json', data_files='data_downloaded_manually/relationships.json', streaming=True)['train']
        else:
            self.dataset = load_dataset('visual_genome', 'relationships_v1.0.0', split=split)
        self.transform = transform

    def __iter__(self):
        return iter(self.dataset)

def pad_collate_fn_objs(batch):
    """
    Custom collate function that:
    1. Extracts `image_id`
    2. Converts `relationships` into tensors
    3. Pads sequences to max length in the batch
    """

    image_ids = [item["image_id"] for item in batch]

    objects_list = [item["objects"] for item in batch]

    # Convert relationships to tensor representations (Example: using their lengths)
    object_lengths = torch.tensor([len(obj) for obj in objects_list])  # Just an example

    # If each relationship is a dictionary, we need to extract meaningful features
    max_obj_length = max(object_lengths).item()  # Find max length in batch
    padded_objects = []

    for obj in objects_list:
        # Example: Convert each relation to a fixed-size vector (e.g., one-hot or embeddings)
        # Here, we just store the count of relations in a tensor
        # Note: You can use a more sophisticated method to encode relationships
        obj_tensor = torch.tensor([len(obj)] + [0] * (max_obj_length - len(obj)))  # Pad with 0s
        padded_objects.append(obj_tensor)

    padded_objects = pad_sequence(padded_objects, batch_first=True, padding_value=0)


    return {
        "image_id": torch.tensor(image_ids),
        "objects": padded_objects,  # Padded tensor
    }    

def pad_collate_fn_rels(batch):
    """
    Custom collate function that:
    1. Extracts `image_id`
    2. Converts `relationships` into tensors
    3. Pads sequences to max length in the batch
    """

    image_ids = [item["image_id"] for item in batch]

    relationships_list = [item["relationships"] for item in batch]

    # Convert relationships to tensor representations (Example: using their lengths)
    relationship_lengths = torch.tensor([len(rel) for rel in relationships_list])  # Just an example

    # If each relationship is a dictionary, we need to extract meaningful features
    max_rel_length = max(relationship_lengths).item()  # Find max length in batch
    padded_relationships = []

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
    }

def load_visual_genome_objects_data(batch_size=32, shuffle=True, split='train', num_workers=4, transform=None):
    objects = VisualGenomeObjectsDataset(split=split, transform=transform)
    dataloader = DataLoader(objects, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_collate_fn_objs)
    return dataloader

def load_visual_genome_relations_data(batch_size=32, shuffle=True, split='train', num_workers=4, transform=None):
    relationships = VisualGenomeRelationsDataset(split=split, transform=transform)
    dataloader = DataLoader(relationships, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_collate_fn_rels)
    return dataloader

# Test with smaller datasets
def test_loader_with_small_dataset():
    objects = VisualGenomeObjectsDataset(split='train')
    objects_dataloader = DataLoader(objects, batch_size=2, shuffle=False, num_workers=1, collate_fn=pad_collate_fn_objs)
    
    relationships = VisualGenomeRelationsDataset(split='train')
    relations_dataloader = DataLoader(relationships, batch_size=2, shuffle=False, num_workers=1, collate_fn=pad_collate_fn_rels)
    
    return objects_dataloader, relations_dataloader

if __name__ == "__main__":
    test_loader_with_small_dataset()
