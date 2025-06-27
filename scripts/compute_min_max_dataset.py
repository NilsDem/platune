import os
import gin
import torch
import pickle
import click
import json
import numpy as np
from tqdm import tqdm

from platune.datasets.base import SimpleDataset


DISCRETE_ATTRIBUTES = ['melody', 'melody_processed', 'pitch', 'pitch_processed', 'octave', 'octave_processed', 'onsets', 'dynamics', 'velocity', 'instrument']
CONTINUOUS_ATTRIBUTES = ['rms', 'loudness1s', 'integrated_loudness', 'centroid', 'bandwidth', 'booming', 'sharpness']


def compute_bins(all_attr_continuous, nb_bins, continuous_descriptors):
    all_values = {}
    for i, c in enumerate(continuous_descriptors):
        data = all_attr_continuous[:, i, :].flatten()
        data.sort()
        index = np.linspace(0, len(data) - 1, nb_bins).astype(int)
        values = [data[i] for i in index]
        all_values[c] = values
    return all_values


@click.command()
@click.option('--data_path', default="", help='path to input dataset')
@click.option('--n_bins', default=20, help='number of bins to quantize continuous attributes')
@click.option('-d', '--discrete_var', multiple=True, default=DISCRETE_ATTRIBUTES, help='list of discrete audio descriptors')
@click.option('-c', '--continuous_var', multiple=True, default=CONTINUOUS_ATTRIBUTES, help='list of continuous audio descriptors')
def main(data_path, n_bins, discrete_var, continuous_var):
    discrete_var = list(discrete_var)
    continuous_var = list(continuous_var)

    keys = ["z", "metadata"] + discrete_var + continuous_var

    dataset = SimpleDataset(
        path=data_path,
        keys=keys
    )

    # save keys without nan 
    lmdb_keys = []
    skip_keys = []

    min_max = {}

    discrete_var_count = {}

    all_attr_continuous = []

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        attr_continuous = []

        for k, v in data.items():
            
            if k in ['z', 'metadata', 'midi', 'probs_list']:
                continue

            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            
            if np.isnan(v).any():
                current_key = dataset.keys[i]
                if current_key not in skip_keys:
                    print("Skipping example {str(current_key)} because found NaN values.")
                    skip_keys.append(current_key)
                    continue
            
            # store and count categories for each discrete attributes in order to define the number of gaussians for each control dimension
            if k in discrete_var:
                values, count = np.unique(v, return_counts=True)

                if k not in discrete_var_count:
                    discrete_var_count[k] = {
                        int(val): int(c)
                        for val, c in zip(values, count)
                    }
                else:
                    for i in range(len(values)):
                        if int(values[i]) not in discrete_var_count[k]:
                            discrete_var_count[k][int(values[i])] = int(
                                count[i])
                        else:
                            discrete_var_count[k][int(values[i])] += int(
                                count[i])
                            
            # compute min and max for continuous attributes in order to normalize between -1 and 1
            if k in continuous_var:
                attr_continuous.append(v)       
                min_value = np.min(v)
                max_value = np.max(v)
                if k not in min_max:
                    min_max[k] = {'min': float(min_value), 'max': float(max_value)}
                else:
                    if min_value < min_max[k]['min']:
                        min_max[k]['min'] = float(min_value)
                    if max_value > min_max[k]['max']:
                        min_max[k]['max'] = float(max_value)
        
        attr_continuous = np.stack(attr_continuous)
        all_attr_continuous.append(attr_continuous)
                        
    print(min_max)
    print(discrete_var_count)

    all_attr_continuous = np.stack(all_attr_continuous)
    bins_values = compute_bins(all_attr_continuous, n_bins, continuous_var)

    metadata = {
        'data_path': data_path,
        'continuous_attr_min_max': min_max,
        'discrete_attr_var_count': discrete_var_count,
    }

    with open(os.path.join(data_path, 'metadata_attributes.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    with open(os.path.join(data_path, "bins_values.pkl"), "wb") as f:
        pickle.dump(bins_values, f)
    
    print(f"nb skipped examples with nans : {len(skip_keys)}")

    with open(os.path.join(data_path, "skip_keys.pkl"), "wb") as f:
        pickle.dump(skip_keys, f)

    lmdb_keys = [key for key in dataset.keys if key not in skip_keys]

    print(f"nb examples : {len(lmdb_keys)}")

    with open(os.path.join(data_path, "lmdb_keys.pkl"), "wb") as f:
        pickle.dump(lmdb_keys, f)


if __name__ == '__main__':
    main()
