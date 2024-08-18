
#import only the taining and validation dataset by modifying and running prepare data and model, then put models separately or import only them from the same file
import torch
from torch.utils.data import Dataset, DataLoader

from prepare_data_and_model import UNet, count_parameters, train_model, save_model

train_dataset = torch.load('results/train_dataset.pt')
val_dataset = torch.load('results/val_dataset.pt')

import datetime

now = datetime.datetime.now()
print("Current date and time when starting UNet training: ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))



# Settings for training
train_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs':          12,
    'batch_size':        32,
    'learning_rate':     1e-3,
    'batches_per_epoch': 50,
    'lr_decay_factor':   1
}

# Create UNet model and count params
model = UNet()
count_parameters(model)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

# Train model
train_epoch_losses, val_epoch_losses = train_model(model, train_dataloader, val_dataloader, train_config, verbose=False)


torch.save(train_epoch_losses, 'results/train_epoch_losses_UNet.pt')
torch.save(val_epoch_losses, 'results/val_epoch_losses_UNet.pt')


save_model(model, 'results/UNet.pth')


then = datetime.datetime.now()
print("Current date and time : ")
print(then.strftime("%Y-%m-%d %H:%M:%S"))


