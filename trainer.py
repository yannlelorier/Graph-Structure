from visual_gm_data import loader
from src import DRGG_model

def unit_test():
    print('Running data loader test...')
    dataloader = loader.test_loader_with_small_dataset()
    import ipdb; ipdb.set_trace()
    for i, data in enumerate(dataloader):
        print(f"Batch {i+1}: {data}")

    print('Unit test passed')

    print('Running one pass over the network to check if this thing works...')
    print('init model...')
    model = DRGG_model.DRGG_model()
    print('init data loader...')
    objects_dl, relations_dl = loader.test_loader_with_small_dataset()
    print('running one pass over the network...')
    for i, data in enumerate(objects_dl):
        print(f"Batch {i+1}: {data}")
        model(data)
        if i == 10:
            break
    print('Unit test passed')

def train():
    raise NotImplementedError
    # Training code here

if __name__ == "__main__":
    unit_test()

