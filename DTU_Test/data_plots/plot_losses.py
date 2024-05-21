import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print("here:",os.getcwd())

# Import npy file
folder_name = 'DTU_Test/data_plots/TCL_GraphMixer_tgbl-wiki_seed_1_run_0'
all_individual_losses = np.load(f'{folder_name}/all_individual_losses.npy')
all_train_losses = np.load(f'{folder_name}/all_train_losses.npy')
all_train_metrics = np.load(f'{folder_name}/all_train_metrics.npy')
all_val_metrics = np.load(f'{folder_name}/all_val_metric.npy')

# Calculate stats
model_list = folder_name.split('/')[-1].split('-')[0].split('_')[:-1]
num_models = len(model_list)
num_epochs = len(all_train_losses)
len_epoch = len(all_train_losses[0])

# Plot losses
for model, name in enumerate(model_list):
    model_loss = all_individual_losses[:, :, model].flatten()
    plt.plot(model_loss, label=f'{name}')

# Plot ensemble (combined) loss
ensemble_loss = all_train_losses.flatten()
plt.plot(ensemble_loss, label='Ensemble', linestyle='--')

# Plot epochs instead of batches
plt.xticks(np.arange(0, num_epochs*len_epoch, len_epoch), np.arange(0, num_epochs))

plt.xlim(0, num_epochs*len_epoch)
plt.ylim(0)
plt.grid()
plt.legend()
plt.title('Ensemble and Individual Losses')
plt.xlabel('n_Batch')
plt.ylabel('Loss')
plt.show()