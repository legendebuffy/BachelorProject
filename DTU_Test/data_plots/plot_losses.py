import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Plot losses
for model, name in enumerate(model_list):
    # if name != 'TCL':
    #     continue
    model_loss = all_individual_losses[:, :, model].flatten()
    plt.plot(model_loss, label=f'{name}')

ensemble_loss = all_train_losses.flatten()
plt.plot(ensemble_loss, label='Ensemble', linestyle='--')

# plot vertical lines for epochs
for i in range(1, num_epochs):
    plt.axvline(x=i*len_epoch-0.5, color='gray', linestyle='--', alpha=0.4, label='(epochs)' if i==1 else None)

plt.legend()
plt.title('Losses: TCL + GraphMixer (loss not standardized)')
plt.xlabel('n_Batch')
plt.ylabel('Loss')
plt.show()