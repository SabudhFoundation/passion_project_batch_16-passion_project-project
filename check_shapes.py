import pickle

for split in ['train', 'valid', 'test']:
    with open(f'features/{split}_lbp_features.pkl', 'rb') as f:
        X, y = pickle.load(f)
        print(f'{split}: {X.shape}')