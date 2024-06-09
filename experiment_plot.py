#_____________________________________- CREATE dict, POPULATE WITH INDIVIDUAL -______________________________________#
# Dictionary to hold sub-dictionaries for each model group
columns = ["TGN", "DyRep", "DyGForm", "GraphMix", "TCL", "TGAT", "CAWN", "EdgeB"]

sub_wiki = {
    "TGN": (0.4854, 0.4059, 0.9007),
    "DyRep": (0.1238, 0.0844, 0.7179),
    "DyGForm": (0.8162, 0.7828, 0.9627),
    "GraphMix": (0.4624, 0.3852, 0.9007),
    "TCL": (0.5069, 0.4312, 0.9023),
    "TGAT": (0.4927, 0.4146, 0.8985),
    "CAWN": (0.7353, 0.6843, 0.9533),
    "EdgeB": (0.5179, 0.7014, 0.7865),
}

sub_review = {
    "TGN": (0.4029, 0.3479, 0.7872),
    "DyRep": (0.4012, 0.1547, 0.3827),
    "DyGForm": (0.6495, 0.5123, 0.9910),
    "GraphMix": (0.5120, 0.4753, 0.8364),
    "TCL": (0.3866, 0.3143, 0.8758),
    "TGAT": (0.3678, 0.2921, 0.8359),
    "CAWN": (0.4580, 0.3959, 0.8823),
    "EdgeB": (0.0200, 0.2764, 0.4812)
}

sub_coin = {
    "TGN": (0.4315, 0.3625, 0.8999),
    "DyRep": (0.3057, 0.2324, 0.8736),
    "DyGForm": (0.6752, 0.5923, 0.9829),
    "GraphMix": (0.4725, 0.4019, 0.9129),
    "TCL": (0.4626, 0.3896, 0.9169),
    "TGAT": (0.3930, 0.3237, 0.8822),
    "CAWN": (0.3774, 0.3200, 0.8517),
    "EdgeB": (0.1529, 0.4237, 0.6418)
}

sub_comment = {
    "TGN": (0.3901, 0.3185, 0.8591),
    "DyRep": (0.1317, 0.0811, 0.7932),
    "DyGForm": (0.4919, 0.4317, 0.9337),
    "GraphMix": (0.4489, 0.3663, 0.9378),
    "TCL": (0.4183, 0.3529, 0.8584),
    "TGAT": (0.4098, 0.3508, 0.7937),
    "CAWN": (0.2535, 0.2069, 0.8194),
    "EdgeB": (0.0637, 0.2312, 0.4698)
}

sub_flight = {
    "TGN": (0.1797, 0.1199, 0.8172),
    "DyRep": (0.1451, 0.0943, 0.8011),
    "DyGForm": (0.2709, 0.2168, 0.8775),
    "GraphMix": (0.1715, 0.0690, 0.7564),
    "TCL": (0.1110, 0.0690, 0.7564),
    "TGAT": (0.1233, 0.0782, 0.7680),
    "CAWN": (0.2497, 0.2069, 0.8071),
    "EdgeB": (0.0665, 0.3386, 0.6348)
}

# Dictionary to hold all sub-dictionaries
all_data = {
    "sub-wiki": sub_wiki,
    "sub-review": sub_review,
    "sub-coin": sub_coin,
    "sub-comment": sub_comment,
    "sub-flight": sub_flight
}

#_____________________________________- Append UNFROZEN Data -______________________________________#
# append the unfrozen data from runs
new_data = {
    "sub-wiki": {
        "EdgeB_TGN": (0.8930, 0.5711, 0.9068),
        "EdgeB_DyRep": (0.8372, 0.4480, 0.7481),
        "EdgeB_DyGF": (0.9059, 0.7568, 0.9522),
        "EdgeB_GM": (0.9184, 0.6591, 0.9382),
        "EdgeB_TCL": (0.9160, 0.6388, 0.9322),
        "EdgeB_TGAT": (0.9159, 0.6428, 0.9360),
        "EdgeB_CAWN": (0.9037, 0.7421, 0.9414)
    },
    "sub-review": {
        "EdgeB_DyGF": (0.8262, 0.6358,0.7455),
        "EdgeB_GM": (0.5005, 0.4745, 0.6762),
        "EdgeB_TCL": (0.3994, 0.4147, 0.6890),
        "EdgeB_TGAT": (0.3633, 0.4012, 0.6730),
        "EdgeB_CAWN": (0.4458, 0.4527, 0.6180)
    },
    "sub-coin": {
        "EdgeB_TGN": (0.4603, 0.4038, 0.7919),
        "EdgeB_DyRep": (0.4224,0.3551,0.6929),
        "EdgeB_DyGF": (0.6422, 0.5231, 0.8331),
        "EdgeB_GM": (0.5075, 0.4267, 0.7966),
        "EdgeB_TCL": (0.4894, 0.4122, 0.7976),
        "EdgeB_TGAT": (0.4629, 0.3790, 0.7863),
        "EdgeB_CAWN": (0.5400, 0.4679, 0.8043)
    },
    "sub-comment": {
        "EdgeB_DyRep": (0.0474, 0.1692, 0.4791),
        "EdgeB_DyGF": (0.4981, 0.4116, 0.7236),
        "EdgeB_GM": (0.4483, 0.3389, 0.7171),
        "EdgeB_TCL": (0.4176, 0.3390, 0.7002),
        "EdgeB_TGAT": (0.4095, 0.3437, 0.6682),
        "EdgeB_CAWN": (0.3498, 0.3323, 0.6817)
    },
    "sub-flight": {
        "EdgeB_TGN": (0.2461, 0.2146, 0.7211),
        "EdgeB_DyRep": (0.2300, 0.1942, 0.6644),
        "EdgeB_DyGF": (0.3190, 0.2871, 0.7575),
        "EdgeB_GM": (0.2608, 0.2269, 0.7352),
        "EdgeB_TCL": (0.2403, 0.2041, 0.7121),
        "EdgeB_TGAT": (0.2447, 0.2101, 0.7238),
        "EdgeB_CAWN": (0.3181, 0.2823, 0.7402)
    }
}

# Append to all_data
for category, updates in new_data.items():
    all_data[category].update(updates)

#_____________________________________- Append FROZEN Data -______________________________________#
# append the frozen data from runs
new_combined_data = {
    "sub-wiki": {
        "F_EdgeB_TGN": (0.7172, 0.6788, 0.9315),
        "F_EdgeB_DyRep": (0.5390, 0.5094, 0.8413),
        "F_EdgeB_DyGF": (0.8164, 0.7831, 0.9629),
        "F_EdgeB_GM": (0.6645, 0.6237, 0.9207),
        "F_EdgeB_TCL": (0.7338, 0.6979, 0.9313),
        "F_EdgeB_TGAT": (0.6808, 0.6417, 0.9141),
        "F_EdgeB_CAWN": (0.7588, 0.7145, 0.9542)
    },
    "sub-review": {
        "F_EdgeB_TGN": (0.3934, 0.3376, 0.7769),
        "F_EdgeB_DyRep": (0.3837, 0.3190, 0.8591),
        "F_EdgeB_DyGF": (0.4615, 0.3496, 0.9551),
        "F_EdgeB_GM": (0.4330, 0.3885, 0.8064),
        "F_EdgeB_TCL": (0.3647, 0.2948, 0.8630),
        "F_EdgeB_TGAT": (0.3372, 0.2644, 0.8205),
        "F_EdgeB_CAWN": (0.4318, 0.3729, 0.8721)
    },
    "sub-coin": {
        "F_EdgeB_TGN": (0.4396, 0.3770, 0.8549),
        "F_EdgeB_DyRep": (0.3480, 0.2832, 0.8659),
        "F_EdgeB_DyGF": (0.6207, 0.5344, 0.9769),
        "F_EdgeB_GM": (0.4719, 0.4011, 0.9131),
        "F_EdgeB_TCL": (0.4628, 0.3930, 0.9171),
        "F_EdgeB_TGAT": (0.4109, 0.3471, 0.8850),
        "F_EdgeB_CAWN": (0.3798, 0.3266, 0.8389)
    },
    "sub-comment": {
        "F_EdgeB_TGN": (0.3419, 0.2727, 0.8378),
        "F_EdgeB_DyRep": (0.1264, 0.0905, 0.7105),
        "F_EdgeB_DyGF": (0.3202, 0.2755, 0.8332),
        "F_EdgeB_GM": (0.4011, 0.3189, 0.9291),
        "F_EdgeB_TCL": (0.2521, 0.2092, 0.7613),
        "F_EdgeB_TGAT": (0.2496, 0.2109, 0.7013),
        "F_EdgeB_CAWN": (0.1978, 0.1623, 0.7522)
    },
    "sub-flight": {
        "F_EdgeB_TGN": (0.1914, 0.1344, 0.8064),
        "F_EdgeB_DyRep": (0.1693, 0.1184, 0.7970),
        "F_EdgeB_DyGF": (0.2738, 0.2166, 0.8703),
        "F_EdgeB_GM": (0.1861, 0.1317, 0.8054),
        "F_EdgeB_TCL": (0.1441, 0.0977, 0.7724),
        "F_EdgeB_TGAT": (0.1526, 0.1053, 0.7714),
        "F_EdgeB_CAWN": (0.2553, 0.2104, 0.8117)
    }
}

# Append to all_data
for category, updates in new_combined_data.items():
    all_data[category].update(updates)

#_____________________________________- Append Best performing to Data -______________________________________#
# append data from 2 best performing models for each dataset (not review, failed)
new2_combined_data = {
    "sub-wiki": {
        "Best_2": (0.8095,0.4187,0.7349) #DyGF_CAWN
    },
    "sub-coin": {
        "Best_2": (0.6208,0.4704,0.9432) #DyGF_GrMi
    },
    "sub-comment": {
        "Best_2": (0.4124,0.3697,0.8948) #DyGF_GrMi
    },
    "sub-flight": {
        "Best_2": (0.3112,0.2013,0.6984) #DyGF_CAWN
    }
}
for category, updates in new2_combined_data.items():
    all_data[category].update(updates)

# Display the dictionary for sub-wiki as an example
#print(all_data.keys(), all_data["sub-wiki"].keys(), all_data["sub-wiki"]["TGN"][0])

#_____________________________________- PLOTTING -______________________________________#
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# model groups with their variations
base_models = ['TGN', 'DyRep', 'DyGFormer', 'GraphMixer', 'TCL', 'TGAT', 'CAWN', 'EdgeBank', 'Best_comb']
variation_mapping = {
    'TGN': ['TGN', 'F_EdgeB_TGN', 'EdgeB_TGN'],
    'DyRep': ['DyRep', 'F_EdgeB_DyRep', 'EdgeB_DyRep'],
    'DyGFormer': ['DyGForm', 'F_EdgeB_DyGF', 'EdgeB_DyGF'],
    'GraphMixer': ['GraphMix', 'F_EdgeB_GM', 'EdgeB_GM'],
    'TCL': ['TCL', 'F_EdgeB_TCL', 'EdgeB_TCL'],
    'TGAT': ['TGAT', 'F_EdgeB_TGAT', 'EdgeB_TGAT'],
    'CAWN': ['CAWN', 'F_EdgeB_CAWN', 'EdgeB_CAWN'],
    'EdgeBank': ['EdgeB'],
    'Best_comb': ['Best_2']
}

# order of categories
category_order = {
    'Individual': 0,
    'Frozen': 1,
    'Unfrozen': 2
}
# Names for the Best_comb over each dataset
best_comb_names = ['DyG+CAW', 'DyG+GrM', 'DyG+GrM', 'DyG+GrM', 'DyG+CAW']

def plot_data(base_models, metric='MRR', with_individuals=True):

    if metric == 'MRR':
        metric_index = 0

        color_map = {
            'Individual': '#add8e6',    # Light Blue
            'Frozen': '#6495ed',        # Medium Blue
            'Unfrozen': '#00008b'       # Dark Blue
        }

    elif metric == 'PR AUC':
        metric_index = 1

        color_map = {
            'Individual': '#ff9999',    # Light Red
            'Frozen': '#ff4d4d',        # Medium Red
            'Unfrozen': '#b30000'       # Dark Red
        }


    elif metric == 'ROC AUC':
        metric_index = 2

        color_map = {
            'Individual': '#98FB98',    # Light Green
            'Frozen': '#32CD32',        # Medium Green
            'Unfrozen': '#006400'       # Dark Green
        }

    if not with_individuals:
        color_map.pop('Individual')
        category_order.pop('Individual')

        base_models = base_models[:-2]
        for key in variation_mapping:
            variation_mapping[key] = [item for item in variation_mapping[key] if '_' in item]

        keys_to_remove = list(variation_mapping.keys())[-2:]
        for key in keys_to_remove:
            del variation_mapping[key]


    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 20), sharex=True, dpi=115)

    bar_width = 0.15
    gap_width = 0.1  # Standard gap between groups
    extra_gap = 0.25  # Extra gap for last two, Edgebank and Best_comb

    # Iterate over each category and the corresponding axis
    for ax, (category, data) in zip(axes, all_data.items()):
        x_position = 0  # reset position for each category
        tick_positions = [] 

        for base_model in base_models:
            # Sorting variants according to the defined order
            if not with_individuals:
                variants = sorted(variation_mapping.get(base_model, []), 
                  key=lambda v: category_order['Frozen'] if 'F_' in v else category_order['Unfrozen'])
            else:   
                variants = sorted(variation_mapping.get(base_model, []), 
                                key=lambda v: category_order['Individual'] if '_' not in v else category_order['Frozen'] if 'F_' in v else category_order['Unfrozen'])
            positions = [x_position + i * bar_width for i in range(len(variants))]

            # Plot each variant
            for pos, variant in zip(positions, variants):
                if variant in data:
                    value = data[variant][metric_index]  # the metric value

                    if not with_individuals:
                        color = color_map['Frozen'] if 'F_' in variant else color_map['Unfrozen']
                    else:

                        color = color_map['Individual'] if '_' not in variant else color_map['Frozen'] if 'F_' in variant else color_map['Unfrozen']
                    ax.bar(pos, value, bar_width, label=variant, color=color, zorder=3)
            
            # center position for x-tick of this group
            if with_individuals:
                center_position = positions[len(variants) // 2]
            else:
                center_position = (positions[0] + positions[1]) / 2
            tick_positions.append(center_position)
            # Apply extra gap for the last two model groups
            if base_model in ['CAWN','EdgeBank', 'Best_comb']:
                x_position += len(variants) * bar_width + extra_gap
            else:
                x_position += len(variants) * bar_width + gap_width
        
        ax.set_title(f'Data for {category}')
        ax.set_xticks(tick_positions)
        if not with_individuals:
            ax.set_xticklabels(base_models)
        else:
            ax.set_xticklabels(base_models[:-1]+[""])
        ax.set_ylim(0, 1)  # Maybe adjust?
        ax.yaxis.grid(True, zorder=0)
        ax.set_yticks([i/4 for i in range(0, 4)])

    # Hardcoding the Best_comb labels
    if with_individuals:
        for i, custom_name in enumerate(best_comb_names):
            axes[i].text(0.935, -0.19, custom_name, transform=axes[i].transAxes, ha='right')

    axes[-1].set_xlabel('Model Groups')
    fig.supylabel(metric)

    legend_elements = [Patch(facecolor=color, label=label) for label, color in color_map.items()]
    fig.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    fig.subplots_adjust(left=0.072, right=0.94, top=0.966, bottom=0.065, hspace=0.36)
    plt.show()


### Plottin the data
# metrics = MRR, PR AUC, ROC AUC
plot_data(base_models, metric="MRR", with_individuals=False)
