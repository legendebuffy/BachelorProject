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

# stats
model_list = folder_name.split('/')[-1].split('-')[0].split('_')[:-1]
num_models = len(model_list)
num_epochs = len(all_train_losses)
len_epoch = len(all_train_losses[0])

fig, ax = plt.subplots(2,1, figsize=(12, 10))
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

# Plot losses
for model, name in enumerate(model_list):
    model_loss = all_individual_losses[:, :, model].flatten()
    ax[0].plot(model_loss, label=f'{name}')
ensemble_loss = all_train_losses.flatten()
ax[0].plot(ensemble_loss, label='Ensemble', linestyle='--')
ax[0].set_xticks(np.arange(0, num_epochs*len_epoch, len_epoch))
ax[0].set_xticklabels(np.arange(0, num_epochs))
ax[0].set_xlim(0, num_epochs*len_epoch)
ax[0].set_ylim(0)
ax[0].grid()
ax[0].legend()
ax[0].set_title('Losses: TCL + GraphMixer (loss equally weighted)')
ax[0].set_xlabel('n_Batch')
ax[0].set_ylabel('Loss')

# Plot epoch mrr
mrr = all_val_metrics
ax[1].plot(mrr, label='MRR')
ax[1].set_xticks(np.arange(0, num_epochs, 1))
ax[1].set_xticklabels(np.arange(0, num_epochs))
ax[1].set_xlim(0, num_epochs-1)
ax[1].grid()
ax[1].legend()
ax[1].set_title('Validation MRR')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('MRR')

plt.show()
