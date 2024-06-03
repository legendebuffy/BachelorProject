import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import npy file
# folder_name = 'saved_results/DyGFormer_EdgeBank/tgbl-comment/w_02_bs_1500'
folder_name = 'saved_results/DyGFormer_TCL/tgbl-wiki/w_08'

all_train_losses = np.load(f'{folder_name}/all_train_losses.npy')
all_train_metrics = np.load(f'{folder_name}/all_train_metrics.npy')
all_val_metrics = np.load(f'{folder_name}/all_val_metric.npy')


bool_plot_ensemble = 'all_individual_losses.npy' in os.listdir(folder_name)

if bool_plot_ensemble:
    all_individual_losses = np.load(f'{folder_name}/all_individual_losses.npy')
    model_list = folder_name.split('/')[-3].split('_')
    num_models = len(model_list)
    ensemble_coefficients = np.load(f'{folder_name}/ensemble_coefficients.pkl', allow_pickle=True)
else:
    model_name = folder_name.split('/')[-3]
all_val_pr = np.load(f'{folder_name}/all_val_pr.npy')
all_val_roc = np.load(f'{folder_name}/all_val_roc.npy')
data_name = folder_name.split('/')[-2]
num_epochs = len(all_train_losses)
len_epoch = len(all_train_losses[0])
run_name = folder_name.split('/')[-1]

fig, ax = plt.subplots(2, 2, figsize=(15, 15)) 
fig.suptitle(f"{model_name if not bool_plot_ensemble else '+'.join(model_list)}, {data_name}, ({run_name})", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.2)  # Adjusted spacing
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

# Plot 3: Average Precision
# if not bool_plot_ensemble:
ax[1, 0].plot(all_val_pr, label='PR AUC (validation)')
ax[1, 0].plot(all_val_roc, label='ROC AUC (validation)')
ax[1, 0].plot(all_train_metrics[0], label='PR AUC (training)')
ax[1, 0].plot(all_train_metrics[1], label='ROC AUC (training)')
ax[1, 0].set_ylim(0, 1.1*max(max(e) for e in [all_val_pr, all_val_roc] + [i for i in all_train_metrics]))

ax[1, 0].set_xticks(np.arange(0, num_epochs, 1))
ax[1, 0].set_xticklabels(np.arange(1, num_epochs+1))
ax[1, 0].set_xlim(0, num_epochs-1)
ax[1, 0].grid()
ax[1, 0].legend()
ax[1, 0].set_title('Validation AUC metrics')
ax[1, 0].set_xlabel('Epoch')
ax[1, 0].set_ylabel('AUC score')

# ensemble coeff
if bool_plot_ensemble:
    for key, value in ensemble_coefficients.items():
        if key == 'bias':
            continue
        ax[1, 1].plot(value, label=key)
    ax[1, 1].plot(ensemble_coefficients['bias'], label='bias')
    # ax[1, 1].set_ylim(0, 1.1*max([max(e) for e in ensemble_coefficients.values()]))
    ax[1, 1].set_xticks(np.arange(0, num_epochs, 1))
    ax[1, 1].set_xticklabels(np.arange(1, num_epochs+1))
    ax[1, 1].set_xlim(0, num_epochs-1)
    ax[1, 1].grid()
    ax[1, 1].legend()
    ax[1, 1].set_title('Ensemble combiner coefficients')
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_ylabel('Coefficient')

plt.show()