import os
import numpy as np
from path import DATA_PATH

audio_path = []
datasets = os.listdir(DATA_PATH)
for dataset in datasets:
    if not os.path.isdir(os.path.join(DATA_PATH, dataset)):
        continue
    mixes = os.listdir(os.path.join(DATA_PATH, dataset))
    for mix in mixes:
        files = os.listdir(os.path.join(DATA_PATH, dataset, mix))
        for file in files:
            if file[-4:] != 'flac':
                continue
            audio_path.append(os.path.join(DATA_PATH, dataset, mix, file))

np.random.seed(2024)
selected_files = np.random.choice(audio_path, 50, replace=False)
print(f'we select 50 out of {len(audio_path)} files as the training data')

with open(os.path.join(DATA_PATH, "cae_training_filelist.txt"), "w") as f:
    for file in selected_files:
        f.write(file + "\n")
