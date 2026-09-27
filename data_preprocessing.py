import os
import pandas as pd
import numpy as np
import torch
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

ZONE_MAP = {
    'q':1, 'w':1, 'e':1, 'r':1, 't':1, 'a':1, 's':1, 'd':1, 'f':1, 'g':1, 'z':1, 'x':1, 'c':1, 'v':1, 'b':1,
    'y':2, 'u':2, 'i':2, 'o':2, 'p':2, 'h':2, 'j':2, 'k':2, 'l':2, 'n':2, 'm':2,
    'space':3, 'enter':4, 'backspace':5, 'shift':6
}

le = LabelEncoder()
all_possible_transitions = [f"{z1}_{z2}" for z1 in range(1,7) for z2 in range(1,7)]
le.fit(all_possible_transitions)
GLOBAL_VOCAB_SIZE = len(le.classes_)

def extract_features(df):
    df = df.sort_values(by='Press_Time').reset_index(drop=True)
    df['Zone'] = df['Key'].str.lower().map(ZONE_MAP).fillna(0).astype(int)
    
    features = []
    for i in range(len(df) - 1):
        curr_row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        dwell_time = curr_row['Release_Time'] - curr_row['Press_Time']
        flight_time = next_row['Press_Time'] - curr_row['Release_Time']
        transition = f"{curr_row['Zone']}_{next_row['Zone']}"
        
        try:
            trans_encoded = le.transform([transition])[0]
            features.append([dwell_time, flight_time, trans_encoded])
        except ValueError:
            continue
            
    return np.array(features)

class KeystrokeChunkDataset(Dataset):
    def __init__(self, data_pairs, labels):
        self.data_pairs = data_pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x1, x2 = self.data_pairs[idx]
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def prepare_chunk(file_paths, seq_length=20):
    valid_data = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['Key', 'Press_Time', 'Release_Time'])
            features = extract_features(df)
            if len(features) >= seq_length:
                valid_data.append(features[:seq_length])
        except:
            continue

    if len(valid_data) < 2: return None

    scaler = StandardScaler()
    all_numeric = np.vstack(valid_data)[:, :2]
    scaler.fit(all_numeric)
    
    for i in range(len(valid_data)):
        valid_data[i][:, :2] = scaler.transform(valid_data[i][:, :2])

    pairs = []
    labels = []
    for _ in range(len(valid_data) * 2):
        if random.random() > 0.5:
            idx = random.randint(0, len(valid_data) - 1)
            pairs.append((valid_data[idx], valid_data[idx]))
            labels.append(1)
        else:
            idx1, idx2 = random.sample(range(len(valid_data)), 2)
            pairs.append((valid_data[idx1], valid_data[idx2]))
            labels.append(0)

    dataset = KeystrokeChunkDataset(pairs, labels)
    return DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)