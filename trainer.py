import torch
from src.DRGG_model import DRGGModel
from visual_gm_data import loader
from src import DRGG_model
from torch.utils.tensorboard import SummaryWriter


def unit_test():
    # print('Running data loader test...')
    # dataloader = loader.test_loader_with_small_dataset()
    # import ipdb; ipdb.set_trace()
    # for i, data in enumerate(dataloader):
    #     print(f"Batch {i+1}: {data}")

    # print('Unit test passed')

    print('Running one pass over the network to check if this thing works...')
    #print('init model...')
    #model = DRGG_model.DRGGModel()
    print('init data loader...')
    objects_dl, _ = loader.test_loader_with_small_dataset()

    i = 0
    print('running one pass over the network...')
    for batch in objects_dl:
        print(batch)
        try:
            assert len(batch) == 2
            assert 'relationships' in batch or 'objects' in batch
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

    criterion = torch.nn.CrossEntropyLoss()

    # load the dataset
    #batch size is 4
    objects_dataset = loader.load_visual_genome_objects_data()
    relations_dataset = loader.load_visual_genome_relations_data()

    for epoch in range(10):
        for objects_batch, relations_batch in zip(objects_dataset, relations_dataset):
            optimizer.zero_grad()
            object_preds, relation_preds = model(objects_batch, relations_batch)
            obj_loss = criterion(object_preds, objects_batch['labels'])
            rel_loss = criterion(relation_preds, relations_batch['labels'])
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
    train()

