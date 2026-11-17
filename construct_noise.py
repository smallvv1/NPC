import argparse
import os

import numpy as np

def make_noise(original_caption, noise_ratio, output_dir=None, seed=None):
    '''
    original_caption: The original caption file path of the dataset. Usually, it is a ".txt" file.
    noise_ratio: The noise proportion you need. (e.g., 0.2, 0.4, and 0.6)
    '''
    if not 0 <= noise_ratio <= 1:
        raise ValueError("noise_ratio must be between 0 and 1")

    if seed is not None:
        np.random.seed(seed)

    captions = []
    for line in open(original_caption, 'r', encoding='utf-8'):
        captions.append(line.strip())
    length = len(captions)
    idx = np.arange(length)
    np.random.shuffle(idx)
    # get noise's number
    noise_length = int(noise_ratio * length)
    shuffle_cap= np.array(captions)[idx[:noise_length]]
    np.random.shuffle(shuffle_cap)
    noise_cap = np.array(captions)
    noise_cap[idx[:noise_length]] = shuffle_cap
    if output_dir is None:
        output_dir = os.path.dirname(original_caption)
    os.makedirs(output_dir, exist_ok=True)
    ratio_name = '0' if float(noise_ratio) == 0 else str(noise_ratio)
    output_file = os.path.join(output_dir, '{}_noise_train_caps.txt'.format(ratio_name))
    with open(output_file, mode='w', encoding='utf-8') as f:
            for cap in list(noise_cap):
                f.write(cap + '\n')
    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct a shuffled-caption CC120K noise file.')
    parser.add_argument('--dataset_root', default='/home/data/CC120K')
    parser.add_argument('--noise_ratio', type=float, required=True)
    parser.add_argument('--seed', type=int, default=99)
    args = parser.parse_args()
    output = make_noise(
        os.path.join(args.dataset_root, 'annotations/scan_split/train_caps.txt'),
        args.noise_ratio,
        seed=args.seed,
    )
    print(output)
