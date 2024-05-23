import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import npy file
folder_name = 'DTU_Test/data_plots/individual/DyGFormer/tgbl-wiki/run1'

all_train_losses = np.load(f'{folder_name}/all_train_losses.npy')
all_train_metrics = np.load(f'{folder_name}/all_train_metrics.npy')
all_val_metrics = np.load(f'{folder_name}/all_val_metric.npy')

bool_plot_ensemble = 'all_individual_losses.npy' in os.listdir(folder_name)

if bool_plot_ensemble:
    all_individual_losses = np.load(f'{folder_name}/all_individual_losses.npy')
    model_list = folder_name.split('/')[-3].split('_')
    num_models = len(model_list)
else:
    model_name = folder_name.split('/')[-3]
data_name = folder_name.split('/')[-2]
num_epochs = len(all_train_losses)
len_epoch = len(all_train_losses[0])

fig, ax = plt.subplots(2,1, figsize=(12, 10))
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Plot losses
if bool_plot_ensemble:
    for model, name in enumerate(model_list):
        model_loss = all_individual_losses[:, :, model].flatten()
        ax[0].plot(model_loss, label=f'{name}', alpha=0.9)

ensemble_loss = all_train_losses.flatten()
ax[0].plot(ensemble_loss, label='Ensemble', alpha=0.5 if bool_plot_ensemble else None)
ax[0].set_xticks(np.arange(0, num_epochs*len_epoch, len_epoch))
ax[0].set_xticklabels(np.arange(1, num_epochs+1))
ax[0].set_xlim(0, num_epochs*len_epoch)
ax[0].set_ylim(0)
ax[0].grid()
if bool_plot_ensemble:
    ax[0].set_title(f'Ensemble losses ({"+".join(model_list)}), {data_name}')
    ax[0].legend()
else:
    ax[0].set_title(f'Loss, {model_name}, {data_name}')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')

# Plot epoch mrr
mrr = all_val_metrics
ax[1].plot(mrr, label='MRR')
ax[1].set_xticks(np.arange(0, num_epochs, 1))
ax[1].set_xticklabels(np.arange(1, num_epochs+1))
ax[1].set_xlim(0, num_epochs-1)
ax[1].grid()
if bool_plot_ensemble:
    ax[1].legend()
ax[1].set_title(f'Validation MRR, {"ensemble" if bool_plot_ensemble else model_name}')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('MRR')

plt.show()