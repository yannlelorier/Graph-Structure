from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class VisualGenomeObjectsDataset(Dataset):
    def __init__(self, local=True, split='train', transform=None):
        if local:
            self.dataset = load_dataset('json', data_files='data_downloaded_manually/objects.json', streaming=True)
        else:
            self.dataset = load_dataset('visual_genome', 'objects_v1.0.0', split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if self.transform:
            image = self.transform(image)
        return image

class VisualGenomeRelationsDataset(Dataset):
    def __init__(self, local=True, split='train', transform=None):
        if local:
            self.dataset = load_dataset('json', data_files='data_downloaded_manually/relationships.json', streaming=True)
        else:
            self.dataset = load_dataset('visual_genome', 'relationships_v1.0.0', split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if self.transform:
            image = self.transform(image)
        return image

def load_visual_genome_objects_data(batch_size=32, shuffle=True, split='train', num_workers=4, transform=None):
    dataset = VisualGenomeObjectsDataset(split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def load_visual_genome_relations_data(batch_size=32, shuffle=True, split='train', num_workers=4, transform=None):
    dataset = VisualGenomeRelationsDataset(split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Test with smaller datasets
def test_loader_with_small_dataset():
    objects_dataset = VisualGenomeObjectsDataset(split='train')
    objects_dataloader = DataLoader(objects_dataset, batch_size=2, shuffle=False, num_workers=1)
    
    relations_dataset = VisualGenomeRelationsDataset(split='train')
    relations_dataloader = DataLoader(relations_dataset, batch_size=2, shuffle=False, num_workers=1)
    
    return objects_dataloader, relations_dataloader

if __name__ == "__main__":
    test_loader_with_small_dataset()
