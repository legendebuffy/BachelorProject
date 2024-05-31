import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import npy file
#folder_name = 'DTU_Test/data_plots/individual/DyGFormer/tgbl-wiki/run1'
folder_name = 'saved_results/CAWN/tgbl-flight/CAWN_AH'
#folder_name = 'DTU_Test/data_plots/individual/TGN/tgbl-wiki_VARM'

all_train_losses = np.load(f'{folder_name}/all_train_losses.npy')
all_train_metrics = np.load(f'{folder_name}/all_train_metrics.npy')
all_val_metrics = np.load(f'{folder_name}/all_val_metric.npy')
all_val_pr = np.load(f'{folder_name}/all_val_pr.npy')
all_val_roc = np.load(f'{folder_name}/all_val_roc.npy')

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

fig, ax = plt.subplots(3, 2, figsize=(15, 20))
fig.suptitle(f"{model_name if not bool_plot_ensemble else '+'.join(model_list)}, {data_name}", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(hspace=0.6, wspace=0.2)  # Adjusted spacing
fig.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.1)

# Plot 1: Training Losses
if bool_plot_ensemble:
    for model, name in enumerate(model_list):
        model_loss = all_individual_losses[:, :, model].flatten()
        ax[0, 0].plot(model_loss, label=f'{name}', alpha=0.9)
ensemble_loss = all_train_losses.flatten()
ax[0, 0].plot(ensemble_loss, label=('ensemble ' if bool_plot_ensemble else '')+'loss (per batch)', alpha=0.5)
ensemble_loss_epoch = ensemble_loss.reshape(num_epochs, len_epoch).mean(axis=1)
ax[0, 0].plot(np.arange(0, num_epochs*len_epoch, len_epoch), ensemble_loss_epoch, label=('ensemble ' if bool_plot_ensemble else '')+'loss (epoch avg)', alpha=0.95)
ax[0, 0].set_xticks(np.arange(0, num_epochs*len_epoch, len_epoch))
ax[0, 0].set_xticklabels(np.arange(1, num_epochs+1))
ax[0, 0].set_xlim(0, num_epochs*len_epoch)
ax[0, 0].set_ylim(0, max(1.1*ensemble_loss.max(), (0 if not bool_plot_ensemble else 1.1*max([all_individual_losses[:, :, model].max() for model in range(num_models)]))))
ax[0, 0].grid()
ax[0, 0].legend()
ax[0, 0].set_title(f'Training Loss')
ax[0, 0].set_xlabel('Epoch')
ax[0, 0].set_ylabel('Loss')

# Plot 2: Validation MRR
ax[0, 1].plot(all_val_metrics, label='MRR')
ax[0, 1].set_xticks(np.arange(0, num_epochs, 1))
ax[0, 1].set_xticklabels(np.arange(1, num_epochs+1))
ax[0, 1].set_xlim(0, num_epochs-1)
ax[0, 1].set_ylim(0, 1.1*max(all_val_metrics))
ax[0, 1].grid()
# ax[0, 1].legend()
ax[0, 1].set_title(f'Validation MRR')
ax[0, 1].set_xlabel('Epoch')
ax[0, 1].set_ylabel('MRR')

# Plot 3: PR ROC
ax[1, 0].plot(all_val_pr, label='PR AUC')
ax[1, 0].set_xticks(np.arange(0, num_epochs, 1))
ax[1, 0].set_xticklabels(np.arange(1, num_epochs+1))
ax[1, 0].set_xlim(0, num_epochs-1)
ax[1, 0].set_ylim(0, 1.1*max(all_val_pr))
ax[1, 0].grid()
# ax[1, 0].legend()
ax[1, 0].set_title('Validation PR ROC')
ax[1, 0].set_xlabel('Epoch')
ax[1, 0].set_ylabel('PR ROC')

# Plot 4: ROC AUC
ax[1, 1].plot(all_val_roc, label='ROC AUC')
ax[1, 1].set_xticks(np.arange(0, num_epochs, 1))
ax[1, 1].set_xticklabels(np.arange(1, num_epochs+1))
ax[1, 1].set_xlim(0, num_epochs-1)
ax[1, 1].set_ylim(0, 1.1*max(all_val_roc))
ax[1, 1].grid()
# ax[1, 1].legend()
ax[1, 1].set_title('Validation ROC AUC')
ax[1, 1].set_xlabel('Epoch')
ax[1, 1].set_ylabel('ROC AUC')

# New Plot 5: Training PR AUC
ax[2, 0].plot(all_train_metrics[0], label='Training PR AUC')
ax[2, 0].set_xticks(np.arange(0, num_epochs, 1))
ax[2, 0].set_xticklabels(np.arange(1, num_epochs+1))
ax[2, 0].set_xlim(0, num_epochs-1)
ax[2, 0].set_ylim(0, 1.1*max(all_train_metrics[0]))
ax[2, 0].grid()
ax[2, 0].set_title('Training PR AUC')
ax[2, 0].set_xlabel('Epoch')
ax[2, 0].set_ylabel('PR AUC')

# New Plot 6: Training ROC AUC
ax[2, 1].plot(all_train_metrics[1], label='Training ROC AUC')
ax[2, 1].set_xticks(np.arange(0, num_epochs, 1))
ax[2, 1].set_xticklabels(np.arange(1, num_epochs+1))
ax[2, 1].set_xlim(0, num_epochs-1)
ax[2, 1].set_ylim(0, 1.1*max(all_train_metrics[1]))
ax[2, 1].grid()
ax[2, 1].set_title('Training ROC AUC')
ax[2, 1].set_xlabel('Epoch')
ax[2, 1].set_ylabel('ROC AUC')

plt.show()