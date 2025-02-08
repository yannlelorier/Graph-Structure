from visual_gm_data import loader
from src import DRGG_model
from datasets import load_dataset


from src.DRGG_model import DRGGModel  
from visual_gm_data.loader import get_data_loader  
from torch.utils.tensorboard import SummaryWriter  
import torch


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
    # Initialisation de TensorBoard
    writer = SummaryWriter(log_dir="runs/experiment1")

    # Charger les données
    train_loader, val_loader = get_data_loader()

    # Initialiser le modèle
    model = DRGGModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Entraînement
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log dans TensorBoard
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()}")

        # Validation après chaque epoch
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        writer.add_scalar("Accuracy/val", accuracy, epoch)
        print(f"Validation Accuracy: {accuracy:.4f}")

    # Sauvegarder le modèle
    torch.save(model.state_dict(), "drgg_model.pth")

    # Fermer TensorBoard
    writer.close()

if __name__ == "__main__":
    unit_test()

