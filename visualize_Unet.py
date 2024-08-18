#capire come si ricava il validation set, guarda meglio dati di input
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from prepare_data_and_model import UNet2, plot_learning_curves, display_test_sample, save_image

train_epoch_losses = torch.load('results_initial/train_epoch_losses.pt')
val_epoch_losses = torch.load('results_initial/val_epoch_losses.pt')
val_dataset = torch.load('results/val_dataset.pt')


test_input_iterator = iter(DataLoader(val_dataset, batch_size=1, shuffle=False))

#learning curve 
plot_learning_curves(train_epoch_losses, val_epoch_losses)



# Set so model and these images are on the same device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
# Get an image from the validation dataset that the model hasn't been trained on
test_input, test_target = next(test_input_iterator)

model = UNet2()
model.load_state_dict(torch.load('results_initial/UNet.pth'))

display_test_sample(model, test_input, test_target, device)

# name your Pdf file  CHANGE WITH APPROPRIATE NAME
filename = "plots/multi_plot_image_UNet.pdf" 
# call the function defined in prepare_data_and_model.py
save_image(filename) 












