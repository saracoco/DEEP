#capire come si ricava il validation set, guarda meglio dati di input
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from prepare_data_and_model import AttentionVisUNet, plot_learning_curves, display_test_sample, save_image

train_epoch_losses = torch.load('results/train_epoch_losses_AttentionUNet.pt')
val_epoch_losses = torch.load('results/val_epoch_losses_AttentionUNet.pt')
val_dataset = torch.load('results/val_dataset.pt')


test_input_iterator = iter(DataLoader(val_dataset, batch_size=1, shuffle=False))

import datetime

now = datetime.datetime.now()
print("Current date and time in Visualizing Attention UNet attention mask: ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))



device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_input, test_target = next(test_input_iterator)


# Build attention visualisation model, load to device then load trained weights
att_vis_unet = AttentionVisUNet().to(device)
att_vis_unet.load_state_dict(torch.load('results/attention_unet.pth'))

# Forward pass to get attention maps
_, map_4, map_3, map_2, map_1 = att_vis_unet(test_input.to(device))

# Plot atttention maps
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for idx, (ax, att_map) in enumerate(zip(axes.flatten(), [map_4, map_3, map_2, map_1])):
    power = 2**(3 - idx)
    ax.imshow(att_map.detach().cpu().numpy().reshape(120//power,120//power), cmap='magma')
    ax.set_title(f'Attention Map {4-idx}')
    ax.grid(False)

plt.savefig('plots/plot_attention_maps.png')


then = datetime.datetime.now()
print("Current date and time : ")
print(then.strftime("%Y-%m-%d %H:%M:%S"))
