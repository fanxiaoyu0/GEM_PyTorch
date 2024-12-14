import numpy as np
import pandas as pd
import json
import pickle as pkl
import random
from tqdm import tqdm, trange
import time
import os
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from GeoGNN import GeoGNN, calculate_molecule_meta_data_by_smiles

def set_random_seed(random_seed=1024):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # for reproducibility, but slow
    # torch.use_deterministic_algorithms(True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def archive_code(trial_version):
    current_file_path = os.path.abspath(sys.argv[0])
    current_file_name = os.path.basename(current_file_path)
    os.system(f"cp {current_file_path} ../archive/{trial_version}_{current_file_name}")

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)
    
def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)

def multi_process_worker_for_construct_smiles_to_molecule_meta_data_dict(smiles):    
    try:
        molecule_meta_data = calculate_molecule_meta_data_by_smiles(smiles)
    except:
        molecule_meta_data = None
    return smiles, molecule_meta_data

def construct_smiles_to_molecule_meta_data_dict():
    df = pd.read_csv(f"../data/raw/lipophilicity.csv")
    smiles_list = df["smiles"].tolist()
    print("smiles count before deduplication:", len(smiles_list))
    smiles_list = list(set(smiles_list))
    print("smiles count after deduplication:", len(smiles_list))

    smiles_to_molecule_meta_data_dict = {}
    invalid_smiles_list = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        for result in tqdm(executor.map(multi_process_worker_for_construct_smiles_to_molecule_meta_data_dict, [\
                smiles for smiles in smiles_list]), total=len(smiles_list)):
            smiles, molecule_meta_data = result
            if molecule_meta_data is None:
                invalid_smiles_list.append(smiles)
            else:
                smiles_to_molecule_meta_data_dict[smiles] = molecule_meta_data
    save_pickle_file(f'../data/intermediate/smiles_to_molecule_meta_data_dict.pkl', smiles_to_molecule_meta_data_dict)
    print('valid smiles count:', len(smiles_to_molecule_meta_data_dict))
    print('invalid smiles count:', len(invalid_smiles_list))

def construct_data_list():
    df = pd.read_csv(f"../data/raw/lipophilicity.csv")
    smiles_to_molecule_meta_data_dict = load_pickle_file(f'../data/intermediate/smiles_to_molecule_meta_data_dict.pkl')
    data_list = []
    for index, row in df.iterrows():
        smiles = row["smiles"]
        if smiles in smiles_to_molecule_meta_data_dict:
            data_item = {
                "smiles": smiles,
                "molecule_meta_data": smiles_to_molecule_meta_data_dict[smiles],
                "label": row['label'],
                "dataset_type": row["dataset_type"],
            }
            data_list.append(data_item)
    save_pickle_file(f'../data/intermediate/data_list.pkl', data_list)

# TODO: InMemoryDataset?
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        atom_feature_name_list = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
        bond_feature_name_list = ["bond_dir", "bond_type", "is_in_ring"]
        self.data_list = data_list
        for data_item in self.data_list:
            molecule_meta_data = data_item['molecule_meta_data']
            for feature_name in atom_feature_name_list + bond_feature_name_list:
                molecule_meta_data[feature_name] = torch.tensor(molecule_meta_data[feature_name], dtype=torch.long, device='cuda')
            molecule_meta_data['bond_length'] = torch.tensor(molecule_meta_data['bond_length'], dtype=torch.float, device='cuda')
            molecule_meta_data['bond_angle'] = torch.tensor(molecule_meta_data['bond_angle'], dtype=torch.float, device='cuda')
            molecule_meta_data['edges'] = torch.tensor(molecule_meta_data['edges'], dtype=torch.long, device='cuda').T
            molecule_meta_data['BondAngleGraph_edges'] = torch.tensor(molecule_meta_data['BondAngleGraph_edges'], dtype=torch.long, device='cuda').T

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

def collate_fn(data_batch):
    feature_name_to_value_batch_dict = {
        'atomic_num': torch.tensor([], dtype=torch.long, device='cuda'),
        'formal_charge': torch.tensor([], dtype=torch.long, device='cuda'),
        'degree': torch.tensor([], dtype=torch.long, device='cuda'),
        'chiral_tag': torch.tensor([], dtype=torch.long, device='cuda'),
        'total_numHs': torch.tensor([], dtype=torch.long, device='cuda'),
        'is_aromatic': torch.tensor([], dtype=torch.long, device='cuda'),
        'hybridization': torch.tensor([], dtype=torch.long, device='cuda'),
        'bond_dir': torch.tensor([], dtype=torch.long, device='cuda'),
        'bond_type': torch.tensor([], dtype=torch.long, device='cuda'),
        'is_in_ring': torch.tensor([], dtype=torch.long, device='cuda'),
        'bond_length': torch.tensor([], dtype=torch.float, device='cuda'),
        'bond_angle': torch.tensor([], dtype=torch.float, device='cuda'),
    }
    for feature_name in feature_name_to_value_batch_dict.keys():
        feature_name_to_value_batch_dict[feature_name] = torch.cat([data_item['molecule_meta_data'][feature_name] for data_item in data_batch], dim=0)

    atom_bond_graph_node_offset = 0
    bond_angle_graph_node_offset = 0
    atom_bond_graph_edge_offset_batch = []
    bond_angle_graph_edge_offset_batch = []
    for data_item in data_batch:
        molecule_meta_data = data_item['molecule_meta_data']
        atom_count = len(molecule_meta_data['atomic_num'])
        bond_count = len(molecule_meta_data['bond_length'])
        atom_bond_graph_edge_offset_batch.append(atom_bond_graph_node_offset)
        bond_angle_graph_edge_offset_batch.append(bond_angle_graph_node_offset)
        atom_bond_graph_node_offset += atom_count
        bond_angle_graph_node_offset += bond_count
    atom_bond_graph_edge_batch = torch.cat([data_item['molecule_meta_data']['edges'] + atom_bond_graph_edge_offset_batch[i] for i, data_item in enumerate(data_batch)], dim=1)
    bond_angle_graph_edge_batch = torch.cat([data_item['molecule_meta_data']['BondAngleGraph_edges'] + bond_angle_graph_edge_offset_batch[i] for i, data_item in enumerate(data_batch)], dim=1)
    atom_bond_graph_id_batch = torch.cat([torch.full((len(data_item['molecule_meta_data']['atomic_num']), ), i, dtype=torch.long, device='cuda') for i, data_item in enumerate(data_batch)], dim=0)
    bond_angle_graph_id_batch = torch.cat([torch.full((len(data_item['molecule_meta_data']['bond_length']), ), i, dtype=torch.long, device='cuda') for i, data_item in enumerate(data_batch)], dim=0)

    graph_batch = feature_name_to_value_batch_dict
    graph_batch['atom_bond_graph_edge'] = atom_bond_graph_edge_batch
    graph_batch['bond_angle_graph_edge'] = bond_angle_graph_edge_batch
    graph_batch['atom_bond_graph_id'] = atom_bond_graph_id_batch
    graph_batch['bond_angle_graph_id'] = bond_angle_graph_id_batch
    label_batch = torch.tensor([data_item['label'] for data_item in data_batch], device='cuda').unsqueeze(1)
    smiles_batch = [data_item['smiles'] for data_item in data_batch]
    return graph_batch, label_batch, smiles_batch

def get_data_loader():
    batch_size = 256
    data_list = load_pickle_file(f'../data/intermediate/data_list.pkl')
    data_list_train = []
    data_list_validate = []
    data_list_test = []
    for data_item in data_list:
        if data_item['dataset_type'] == 'train':
            data_list_train.append(data_item)
        elif data_item['dataset_type'] == 'validate':
            data_list_validate.append(data_item)
        elif data_item['dataset_type'] == 'test':
            data_list_test.append(data_item)
    data_loader_train = DataLoader(MyDataset(data_list_train), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_validate = DataLoader(MyDataset(data_list_validate), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(MyDataset(data_list_test), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader_train, data_loader_validate, data_loader_test

class GEM_Regressor(nn.Module):
    def __init__(self):
        super(GEM_Regressor, self).__init__()
        self.encoder = GeoGNN(batch_size=256)
        self.encoder.load_state_dict(torch.load("../weight/regression.pth", weights_only=True))
        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, graph_batch):
        node_representation, edge_representation, graph_representaion = self.encoder(graph_batch)
        x = self.mlp(graph_representaion)
        return x

def evaluate(model: GEM_Regressor, data_loader: DataLoader):
    model.eval()
    label_predict = torch.tensor([], dtype=torch.float32).cuda()
    label_true = torch.tensor([], dtype=torch.float32).cuda()
    with torch.no_grad():
        for data_batch in data_loader:
            graph_batch, label_true_batch, smiles_batch = data_batch
            label_predict_batch = model(graph_batch)
            label_predict = torch.cat((label_predict, label_predict_batch.detach()), dim=0)
            label_true = torch.cat((label_true, label_true_batch.detach()), dim=0)
    
    label_predict = label_predict.cpu().numpy()
    label_true = label_true.cpu().numpy()
    rmse = round(float(np.sqrt(mean_squared_error(label_true, label_predict))), 3)
    mae = round(float(mean_absolute_error(label_true, label_predict)), 3)
    r2 = round(r2_score(label_true, label_predict), 3)
    metric = {'rmse': rmse, 'mae': mae, 'r2': r2}
    return metric

def plot_metric_evolution_during_training(trial_version, epoch, metric_list_train, metric_list_validate, metric_list_test, learning_rate_list):
    major_metric = 'rmse'
    major_metric_list_train = [metric[major_metric] for metric in metric_list_train]
    major_metric_list_validate = [metric[major_metric] for metric in metric_list_validate]
    major_metric_list_test = [metric[major_metric] for metric in metric_list_test]

    fig, ax1 = plt.subplots()
    fig.set_size_inches(15, 10)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel(major_metric)
    ax1.plot(range(epoch + 1), major_metric_list_train, label='train', color='tab:blue')
    ax1.plot(range(epoch + 1), major_metric_list_validate, label='validate', color='tab:orange')
    ax1.plot(range(epoch + 1), major_metric_list_test, label='test', color='tab:red')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('learning rate')
    ax2.plot(range(epoch + 1), learning_rate_list, label='learning rate', color='tab:green')
    ax2.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax2.legend(loc='upper right')
    plt.savefig(f'../data/result/{trial_version}_metric_evolution.png')

def train(trial_version):
    data_loader_train, data_loader_validate, data_loader_test = get_data_loader()

    model = GEM_Regressor()
    model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15)

    current_best_metric = None
    current_best_epoch = 0

    metric_list_train = []
    metric_list_validate = []
    metric_list_test = []

    for epoch in range(800):
        model.train()
        for data_batch in data_loader_train:
            graph_batch, label_true_batch, smiles_batch = data_batch
            label_predict_batch = model(graph_batch)
            loss: torch.Tensor = criterion(label_predict_batch, label_true_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # metric_train = evaluate(model, data_loader_train)
        metric_validate = evaluate(model, data_loader_validate)
        metric_test = evaluate(model, data_loader_test)
        # metric_list_train.append(metric_train)
        metric_list_validate.append(metric_validate)
        metric_list_test.append(metric_test)

        major_metric_name = 'rmse'
        if epoch == 0 or metric_validate[major_metric_name] < current_best_metric[major_metric_name]:
            current_best_metric = metric_validate
            current_best_epoch = epoch
            torch.save(model.state_dict(), f"../weight/{trial_version}.pth")
        print("=========================================================")
        print("epoch", epoch)
        # print("metric_train", metric_train)
        print("metric_validate", metric_validate)
        print("metric_test", metric_test)
        print('current_best_epoch', current_best_epoch)
        print('current_best_metric', current_best_metric)
        print("=========================================================")

        # if epoch % 10 == 0:
        #     p = multiprocessing.Process(target=plot_metric_evolution_during_training, args=(trial_version, epoch, metric_list_train, metric_list_validate, metric_list_test, learning_rate_list))
        #     p.start()

    metric_list = {
        # 'metric_list_train': metric_list_train, 
        'metric_list_validate': metric_list_validate, 
        'metric_list_test': metric_list_test
    }
    json.dump(metric_list, open(f'../data/seldom/{trial_version}_metric_list.json', 'w'), indent=4)

def get_predict_label(model, data_loader):
    model.eval()
    label_predict = torch.tensor([], dtype=torch.float32).cuda()
    label_true = torch.tensor([], dtype=torch.float32).cuda()
    smiles_list = []
    with torch.no_grad():
        for data_batch in data_loader:
            graph_batch, label_true_batch, smiles_batch = data_batch
            label_predict_batch = model(graph_batch)
            label_predict = torch.cat((label_predict, label_predict_batch.detach()), dim=0)
            label_true = torch.cat((label_true, label_true_batch.detach()), dim=0)
            smiles_list.extend(smiles_batch)
    
    label_predict = label_predict.cpu().numpy()
    label_true = label_true.cpu().numpy()
    return label_true, label_predict, smiles_list

def test(trial_version):
    data_loader_train, data_loader_validate, data_loader_test = get_data_loader()

    model = GEM_Regressor()
    model.load_state_dict(torch.load(f"../weight/{trial_version}.pth", weights_only=True))
    model.cuda()

    metric_train = evaluate(model, data_loader_train)
    metric_validate = evaluate(model, data_loader_validate)
    metric_test = evaluate(model, data_loader_test)

    print("=========================================================")
    print("metric_train", metric_train)
    print("metric_validate", metric_validate)
    print("metric_test", metric_test)
    print("=========================================================")

    label_true_train, label_predict_train, smiles_list_train = get_predict_label(model, data_loader_train)
    label_true_validate, label_predict_validate, smiles_list_validate = get_predict_label(model, data_loader_validate)
    label_true_test, label_predict_test, smiles_list_test = get_predict_label(model, data_loader_test)

    smiles_to_label_and_error_dict = {}
    for i in range(len(smiles_list_test)):
        smiles = smiles_list_test[i]
        label_true = float(label_true_test[i][0])
        label_predict = float(label_predict_test[i][0])
        error = abs(label_true - label_predict)
        smiles_to_label_and_error_dict[smiles] = {
            'label_true': label_true,
            'label_predict': label_predict,
            'error': error
        }
    json.dump(smiles_to_label_and_error_dict, open(f'../data/result/{trial_version}_smiles_to_label_and_error_dict.json', 'w'), indent=4)

    plt.figure(figsize=(28, 8))
    plt.subplot(1, 3, 1)
    plt.plot([-3, 3], [-3, 3], label='y=x', color='tab:green')
    plt.scatter(label_true_train, label_predict_train, s=1, label='train', color='tab:blue')
    plt.xlabel('label_true')
    plt.ylabel('label_predict')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot([-3, 3], [-3, 3], label='y=x', color='tab:green')
    plt.scatter(label_true_validate, label_predict_validate, s=1, label='validate', color='tab:orange')
    plt.xlabel('label_true')
    plt.ylabel('label_predict')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot([-3, 3], [-3, 3], label='y=x', color='tab:green')
    plt.scatter(label_true_test, label_predict_test, s=1, label='test', color='tab:red')
    plt.xlabel('label_true')
    plt.ylabel('label_predict')
    plt.legend()
    plt.savefig(f'../data/result/{trial_version}_scatter.png')
    plt.close()

    return metric_train, metric_validate, metric_test

if __name__ == '__main__':
    trial_version = '0'
    archive_code(trial_version=trial_version)
    set_random_seed(random_seed=1024)
    # construct_smiles_to_molecule_meta_data_dict()
    # construct_data_list()
    for random_seed in range(1024, 1024 + 5):
        set_random_seed(random_seed=random_seed)
        train(trial_version=f'{trial_version}_{random_seed}')
        test(trial_version=f'{trial_version}_{random_seed}')

    metric_train_list = []
    metric_validate_list = []
    metric_test_list = []
    for random_seed in range(1024, 1024 + 5):
        set_random_seed(random_seed=random_seed)
        metric_train, metric_validate, metric_test = test(trial_version=f'{trial_version}_{random_seed}')
        metric_train_list.append(metric_train)
        metric_validate_list.append(metric_validate)
        metric_test_list.append(metric_test)
    print("=========================================================")
    metric_train_mean = {}
    metric_validate_mean = {}
    metric_test_mean = {}
    metric_train_std = {}
    metric_validate_std = {}
    metric_test_std = {}
    for metric_name in ['rmse', 'mae', 'r2']:
        metric_train_mean[metric_name] = round(np.mean([metric[metric_name] for metric in metric_train_list]), 3)
        metric_validate_mean[metric_name] = round(np.mean([metric[metric_name] for metric in metric_validate_list]), 3)
        metric_test_mean[metric_name] = round(np.mean([metric[metric_name] for metric in metric_test_list]), 3)
        metric_train_std[metric_name] = round(np.std([metric[metric_name] for metric in metric_train_list]), 3)
        metric_validate_std[metric_name] = round(np.std([metric[metric_name] for metric in metric_validate_list]), 3)
        metric_test_std[metric_name] = round(np.std([metric[metric_name] for metric in metric_test_list]), 3)
    print("metric_train_mean", metric_train_mean)
    print("metric_validate_mean", metric_validate_mean)
    print("metric_test_mean", metric_test_mean)
    print("=========================================================")
    print("All is well!")
