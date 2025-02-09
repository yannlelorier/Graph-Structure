import torch
from src.DRGG_model import DRGGModel
from visual_gm_data import loader
from visual_gm_data.loader import ImageTransforms
from torch.utils.tensorboard import SummaryWriter

def unit_test():
    # print('Running data loader test...')
    # dataloader = loader.test_loader_with_small_dataset()
    # import ipdb; ipdb.set_trace()
    # for i, data in enumerate(dataloader):
    #     print(f"Batch {i+1}: {data}")

    # print('Unit test passed')

    print('Running one pass over the network to check if this thing works...')
    print('init a model from scratch...')
    model = DRGGModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3, betas=(0.9, 0.999), eps=1e-8)
    obj_loss = torch.nn.CrossEntropyLoss()
    rel_loss = torch.nn.BCEWithLogitsLoss()
    print('init data loader...')
    dataloader = loader.test_loader_with_small_dataset()

    i = 0
    print('Testing forward pass over the network...')
    import ipdb; ipdb.set_trace()
    for batch in dataloader:
        images = batch["images"]
        image_ids = batch["image_ids"]
        objects = batch["objects"]
        relations = batch["relationships"]

        optimizer.zero_grad()
        print('Batch:\n------\tObjects:', objects, '\n------\tRelations:', relations, '\n------\tImages:', images)
        obj_preds, relation_preds = model(images)
        object_loss = obj_loss(obj_preds, objects)
        relations_loss = rel_loss(relation_preds, relations)
        loss = object_loss + relations_loss
        print(f'Loss: {loss.item()}')
        print('Testing backward pass over the network...')
        loss.backward()
        optimizer.step()

        try:
            assert len() == 2
            assert 'relationships' in relations or 'objects' in objects
            i += 1
            if i == 2:
                break
        except AssertionError as e:
            print(f'{e}:\n\tAssertionError, check the data loader.')
            break

    print(f'{i} Unit tests passed.')

def train():
    state_dict = torch.load('model/model.pth')

    # load the model from state_dict
    model = DRGGModel()
    model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters, lr=1e-4, weight_decay=5e-3, betas=(0.9, 0.999), eps=1e-8)
    optimizer.load_state_dict(state_dict['optimizer'])

    criterion_obj = torch.nn.CrossEntropyLoss()
    criterion_rels = torch.nn.BCEWithLogitsLoss()

    #batch size is 4
    # load the dataset
    transforms = ImageTransforms()
    dataloader = loader.load_visual_genome_data(transform=transforms)

    model.train()

    for epoch in range(10):
        for objects_batch, relations_batch, images_batch in dataloader:
            optimizer.zero_grad()
            
            obj_preds, relation_preds = model(images_batch)
            
            obj_loss = criterion_obj(obj_preds, objects_batch)
            rel_loss = criterion_rels(relation_preds, relations_batch)
            loss = obj_loss + rel_loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            # save the model
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, 'model/model.pth')
if __name__ == "__main__":
    unit_test()
    #train()
