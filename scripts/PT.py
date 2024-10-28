# %%
import numpy as np
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
import torch
import yaml
import json

# %%


def load_hdf5_as_tensors(file_path):
    try:
        with h5py.File(file_path, 'r') as hdf:
            showers_data = hdf["showers"][:]
            showers_tensor = torch.tensor(showers_data, dtype=torch.float32)
            return showers_tensor
            # , energy_tensor
    except KeyError as e:
        print(f"Can not find dataset: {e}")
    except Exception as e:
        print(f"Error: {e}")


# %%
def get_top_patterns(target_label: torch.Tensor):
    """
    Identifies unique patterns in the target_label tensor, counts their occurrences,
    and extracts the top n most frequent patterns.

    Args:
        target_label (torch.Tensor): A tensor containing the target labels with shape [batch_size, pattern_length].
        n (int): The number of top frequent patterns to retrieve.

    Returns:
        target_pattern_collection (dict): A dictionary where keys are unique patterns represented as tuples,
                                         and values are their corresponding counts.
        reduced_matrix (torch.Tensor): A tensor containing the top n most frequent patterns with shape [n, pattern_length].
    """
    # Ensure target_label is on CPU for processing
    target_label_cpu = target_label.cpu()
    
    # Find unique patterns and their counts
    unique_patterns, counts = torch.unique(target_label_cpu, dim=0, return_counts=True)
    
    # Convert unique patterns and counts to a dictionary
    target_pattern_collection = {
        tuple(pattern.tolist()): count.item()
        for pattern, count in zip(unique_patterns, counts)
    }
    
    # Sort the counts in descending order and get the sorted indices
    sorted_counts, sorted_indices = counts.sort(descending=True)
    
    # Select the top n patterns based on the sorted indices
    all_sorted_indices = sorted_indices[:len(unique_patterns)]
    all_sorted_patterns = unique_patterns[all_sorted_indices]
    all_sorted_counts = sorted_counts[:len(unique_patterns)]
    
    # Create a reduced matrix containing only the top n patterns
    all_sorted_patterns = all_sorted_patterns
    all_sorted_pattern_collection = {
        tuple(pattern.tolist()): count.item()
        for pattern, count in zip(all_sorted_patterns, all_sorted_counts)
    }
    
    return all_sorted_pattern_collection

def binary_entropy(possibilities):
    non_zero_possibilities = [p for p in possibilities if p != 0]
    binary_entropy=np.sum(-np.array(non_zero_possibilities)*np.log2(np.array(non_zero_possibilities)))
    return binary_entropy

def merge_dicts(A, B):
    all_keys = set(A) | set(B) 
    sorted_keys = sorted(all_keys, key=lambda k: A.get(k, 0.0), reverse=True)
    Sumed_Dict = {key: [A.get(key, 0.0), B.get(key, 0.0)] for key in sorted_keys}
    return Sumed_Dict

def create_timestamped_folder(base_path):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(base_path, current_time)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# %%
def cube_selection(source, layer_range = 1, ring_range = [1,16], radial_range = [1,9]):
    expanded_source =source.view(source.size(0), 45//layer_range, layer_range, 16, 9)
    cube_subset = torch.sum(expanded_source[:, :, :, ring_range[0]-1:ring_range[1], radial_range[0]-1:radial_range[1]], dim=(2,3,4))
    return cube_subset

# %%
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
current_dir = os.path.dirname(os.path.abspath(__file__))
config = yaml.safe_load(open(os.path.join(current_dir, "..", "configs", "PT_config.yaml")))
file_path_test = config["Target_Path"]
file_path_gen = config["Generated_Path"]
z_length = config["z_length"]
ring_range = config["ring_range"]
radial_range = config["radial_range"]

initial_threshold = config["Threshold"]
output_path = config["Output_Path"]
new_folder = create_timestamped_folder(output_path)
logger.info(f"Loading target data from {file_path_test}")
logger.info(f"Loading generated data from {file_path_gen}")
logger.info(f"Subspace size: z_length: {z_length}, ring_range: {ring_range}, radial_range: {radial_range}")
logger.info(f"Initial threshold: {initial_threshold}")
logger.info(f"Output path: {new_folder}")

if config["Threshold_scan"]:
    threshold_min = config["Threshold_scan_Min"]
    threshold_max = config["Threshold_scan_Max"]
    threshold_step = config["Threshold_scan_Step_Length"]
    logger.info(f"Threshold scan: enabled.")
else:
    logger.info(f"Threshold scan: disabled")


# %%
logger.info(f"Loading data...")
xtarget_samples = load_hdf5_as_tensors(file_path_test)
xgen_samples_gen = load_hdf5_as_tensors(file_path_gen)
logger.info(f"Target data shape: {xtarget_samples.shape}")
logger.info(f"Generated data shape: {xgen_samples_gen.shape}")
logger.info(f"Cutting sub-cylinder...")
cube_target = torch.flatten(cube_selection(xtarget_samples, z_length, ring_range, radial_range), start_dim=1)
cube_gen_gen = torch.flatten(cube_selection(xgen_samples_gen, z_length, ring_range, radial_range), start_dim=1)

# %%
cube_label_target = torch.where(cube_target > initial_threshold, 1, 0)
cube_gen_label_gen = torch.where(cube_gen_gen > initial_threshold, 1, 0)
logger.info(f"Calculating patterns...")
target_pattern_collection = get_top_patterns(cube_label_target)
gen_gen_pattern_collection = get_top_patterns(cube_gen_label_gen)

possibilities_target = [value/xtarget_samples.size(0) for value in target_pattern_collection.values()]
possibilities_gen_gen = [value/xgen_samples_gen.size(0) for value in gen_gen_pattern_collection.values()]

summed_dict_gen = merge_dicts(target_pattern_collection, gen_gen_pattern_collection)
organized_possibilities_target =[]
organized_possibilities_gen = []
for key in summed_dict_gen.keys():
    organized_possibilities_target.append(summed_dict_gen[key][0]/xtarget_samples.size(0))
    organized_possibilities_gen.append(summed_dict_gen[key][1]/xgen_samples_gen.size(0))

logger.info(f'The entropy of the target data is {binary_entropy(possibilities_target)}')
logger.info(f'The entropy of the generated data is {binary_entropy(possibilities_gen_gen)}')
logger.info(f'The number of patterns for the target data is {len(possibilities_target)}')
logger.info(f'The number of patterns for the generated data is {len(possibilities_gen_gen)}')



# %%
n_steps = len(summed_dict_gen)
X = np.arange(0, n_steps)
plt.yscale('log')

plt.step(X + 1, organized_possibilities_target[:n_steps], where='post', label='Target', color = 'black')
plt.step(X + 1, organized_possibilities_gen[:n_steps], where='post', label='Gen', alpha=0.9)

plt.xlabel('Pattern ID')
plt.ylabel('Possibility')
plt.xlim(1, n_steps)
plt.legend()
plt.title('Sub-cylinder Patterns Possibility' )
logger.info(f"Saving sub_cylinder_pattern_possibility.png")
plt.savefig(os.path.join(new_folder, "sub_cylinder_pattern_possibility.png"))
plt.close()

# %%
first_n_patterns = config["n_top_patterns"]
if len(summed_dict_gen) >= first_n_patterns:
    n_steps = first_n_patterns
    X = np.arange(0, n_steps)
    # plt.yscale('log')

    plt.step(X + 1, organized_possibilities_target[:n_steps], where='post', label='Target', color = 'black')
    plt.step(X + 1, organized_possibilities_gen[:n_steps], where='post', label='Gen', alpha=0.9)

    plt.xlabel('Pattern ID')
    plt.ylabel('Possibility')
    plt.xlim(1, n_steps)
    plt.legend()
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title(f'Sub-cylinder First {first_n_patterns} Patterns Possibility' )
    logger.info(f"Saving sub_cylinder_first_{first_n_patterns}_patterns_possibility.png")
    plt.savefig(os.path.join(new_folder, f"sub_cylinder_first_{first_n_patterns}_patterns_possibility.png"))
    plt.close()
else:
    logger.warning(f"Less than {first_n_patterns} patterns in the data, can not plot the first {first_n_patterns} patterns, skipping.")

# %%
pattern_occurrence_dict = {}
for key, value in summed_dict_gen.items():
    target, gen = value
    pattern_occurrence_dict[str(key)] = {'target': target, 'gen': gen}
logger.info(f"Saving pattern_occurrence.json")
json.dump(pattern_occurrence_dict, open(os.path.join(new_folder, "pattern_occurrence.json"), "w"), indent=4)


# %%
if config["Threshold_scan"]:
    logger.info(f"Scanning thresholds from {threshold_min} to {threshold_max} with step length {threshold_step}...")
    thresholds = []
    entropies_target = []
    entropies_gen_gen = []
    for threshold in np.arange(threshold_min, threshold_max+threshold_step, threshold_step):
        
        cube_label_target = torch.where(cube_target > threshold, 1, 0)
        cube_gen_label_gen = torch.where(cube_gen_gen > threshold, 1, 0)

        target_pattern_collection = get_top_patterns(cube_label_target)
        gen_gen_pattern_collection = get_top_patterns(cube_gen_label_gen)

        possibilities_target = [value/xtarget_samples.size(0) for value in target_pattern_collection.values()]
        possibilities_gen_gen = [value/xgen_samples_gen.size(0) for value in gen_gen_pattern_collection.values()]

        entropies_target.append(binary_entropy(possibilities_target))
        entropies_gen_gen.append(binary_entropy(possibilities_gen_gen))

        thresholds.append(threshold)
    threshold_scan_result = {'thresholds': thresholds, 'entropies_target': entropies_target, 'entropies_gen_gen': entropies_gen_gen}
    logger.info(f"Saving threshold_scan_result.json")
    json.dump(threshold_scan_result, open(os.path.join(new_folder, "threshold_scan_result.json"), "w"), indent=4)
    plt.plot(thresholds, entropies_target, label='Target')
    plt.plot(thresholds, entropies_gen_gen, label='Gen')
    plt.xlabel('thresholds in MeV')
    plt.xlim(threshold_min, threshold_max)
    plt.ylabel('Entropy')
    plt.legend(loc='best')
    plt.title('Entropy with scanned thresholds')
    logger.info(f"Saving entropy_with_scanned_thresholds.png")
    plt.savefig(os.path.join(new_folder, "entropy_with_scanned_thresholds.png"))
    plt.close()

# %%
logger.info(f"Finished")


