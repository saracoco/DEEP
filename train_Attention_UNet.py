#import only the taining and validation dataset by modifying and running prepare data and model, then put models separately or import only them from the same file
import torch
from torch.utils.data import Dataset, DataLoader

from prepare_data_and_model import AttentionUNet, count_parameters, train_model, save_model

train_dataloader = torch.load('results/train_dataloader.pt')
val_dataloader = torch.load('results/val_dataloader.pt')

import datetime

now = datetime.datetime.now()
print("Current date and time when starting UNet with Attention (Attention_UNet) training: ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))


train_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs':          25,
    'batch_size':        32,
    'learning_rate':     1e-3,
    'batches_per_epoch': 50,
    'lr_decay_factor':   0.93
}

# Create UNet model and count params
model = AttentionUNet()
count_parameters(model)

# Train model
train_epoch_losses, val_epoch_losses = train_model(model, train_dataloader, val_dataloader, train_config, verbose=False)

torch.save(train_epoch_losses, 'results/train_epoch_losses_AttentionUNet.pt')
torch.save(val_epoch_losses, 'results/val_epoch_losses_AttentionUNet.pt')

# Save weights
save_model(model, 'results/attention_unet.pth')


then = datetime.datetime.now()
print("Current date and time : ")
print(then.strftime("%Y-%m-%d %H:%M:%S"))

