from visual_gm_data import loader
from src import DRGG_model
from datasets import load_dataset

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
    raise NotImplementedError
    # Training code here

if __name__ == "__main__":
    unit_test()

