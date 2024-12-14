import numpy as np
import pdb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool
from line_profiler import profile

RDLogger.DisableLog('rdApp.*')

class RBF(nn.Module):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, dtype=torch.float32):
        super(RBF, self).__init__()
        # Convert centers to a tensor and reshape them
        self.centers = torch.tensor(centers, dtype=dtype, device='cuda').view(1, -1)
        self.gamma = gamma
    
    def forward(self, x):
        """
        Args:
            x (tensor): (-1, 1).
        Returns:
            y (tensor): (-1, n_centers)
        """
        x = x.view(-1, 1)
        return torch.exp(-self.gamma * (x - self.centers) ** 2)

class AtomEmbedding(nn.Module):
    def __init__(self, atom_names, embed_dim):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names
        from rdkit.Chem import ChiralType, HybridizationType
        atom_feature_name_to_possible_value_list_dict = {
            "atomic_num": list(range(1, 119)) + ['misc'],
            "chiral_tag": [
                ChiralType.CHI_UNSPECIFIED, 
                ChiralType.CHI_TETRAHEDRAL_CW, 
                ChiralType.CHI_TETRAHEDRAL_CCW, 
                ChiralType.CHI_OTHER
            ],
            "degree": list(range(11)) + ['misc'],
            "formal_charge": list(range(-5, 11)) + ['misc'],
            "hybridization": [
                HybridizationType.UNSPECIFIED,
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
                HybridizationType.OTHER
            ],
            "is_aromatic": [0, 1],
            "total_numHs": list(range(9)) + ['misc'],
        }

        self.embed_list = nn.ModuleList()
        for name in self.atom_names:
            embed = nn.Embedding(
                    len(atom_feature_name_to_possible_value_list_dict[name]) + 5,
                    embed_dim)
            # 使用 Xavier 初始化方法
            init.xavier_uniform_(embed.weight)
            self.embed_list.append(embed)

    def forward(self, node_features):
        """
        Args: 
            node_features(dict of tensor): node features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[name])
        return out_embed

class BondEmbedding(nn.Module):
    """
    Bond Encoder
    """
    def __init__(self, bond_names, embed_dim):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names
        bond_feature_name_to_possible_value_list_dict = {
            "bond_dir": list(Chem.BondDir.values.values()),
            "bond_type": list(Chem.BondType.values.values()),
            "is_in_ring": [0, 1],
        }

        self.embed_list = nn.ModuleList()
        for name in self.bond_names:
            embed = nn.Embedding(
                len(bond_feature_name_to_possible_value_list_dict[name]) + 5,
                embed_dim
            )
            # 使用 Xavier 初始化
            init.xavier_uniform_(embed.weight)
            self.embed_list.append(embed)

    def forward(self, edge_features):
        """
        Args: 
            edge_features(dict of tensor): edge features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[name])
        return out_embed

class BondFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (np.arange(0, 2, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args: 
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed

class BondAngleFloatRBF(nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (np.arange(0, np.pi, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_angle_float_features):
        """
        Args: 
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            x = bond_angle_float_features[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed

class GINLayer(MessagePassing):
    def __init__(self, hidden_size):
        super(GINLayer, self).__init__(aggr='add')  # "Add" aggregation as in original GIN
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: Node feature matrix with shape [num_nodes, in_channels]
        # edge_index: Graph connectivity (COO format) with shape [2, num_edges]
        # edge_attr: Edge feature matrix with shape [num_edges, in_channels]
        
        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) # time consuming >90%, 1.7s

    def message(self, x_j, edge_attr):
        # x_j: Node features of source nodes
        # edge_attr: Features of edges
        return x_j + edge_attr

    def update(self, aggr_out):
        # aggr_out: Aggregated message
        return self.mlp(aggr_out)

class GraphNorm(nn.Module):
    """Graph Normalization layer for PyTorch Geometric graphs."""
    def __init__(self, batch_size):
        super(GraphNorm, self).__init__()
        self.batch_size = batch_size

    def forward(self, x, batch):
        deg = degree(batch, num_nodes=self.batch_size, dtype=x.dtype).clamp(min=1)
        norm = deg[batch].sqrt()
        return x / norm.view(-1, 1)

class GeoGNNBlock(nn.Module):
    def __init__(self, embed_dim, dropout_rate, last_act, batch_size):
        super(GeoGNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GINLayer(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_norm = GraphNorm(batch_size)
        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    @profile
    def forward(self, x, edge_index, edge_attr, batch):
        """
        x: Node feature matrix
        edge_index: Graph connectivity (COO format)
        edge_attr: Edge feature matrix
        batch: Batch vector which maps each node to its respective graph in the batch
        """
        out = self.gnn(x, edge_index, edge_attr) # time consuming 35%, 0.9s -> 85%, 1.7s
        out = self.norm(out)
        out = self.graph_norm(out, batch) # time consuming 60% -> 10%, 1.6s -> 10%, 0.2s
        if self.last_act:
            out = self.act(out)
        out = self.dropout(out)
        out = out + x
        return out

class GeoGNN(nn.Module):
    def __init__(self, batch_size):
        super(GeoGNN, self).__init__()
        self.embed_dim = 32
        self.dropout_rate = 0.5
        self.layer_num = 8
        self.readout = 'mean'
        self.atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
        self.bond_names = ["bond_dir", "bond_type", "is_in_ring"]
        self.bond_float_names = ["bond_length"]
        self.bond_angle_float_names = ["bond_angle"]

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)
        
        self.bond_embedding_list = nn.ModuleList()
        self.bond_float_rbf_list = nn.ModuleList()
        self.bond_angle_float_rbf_list = nn.ModuleList()
        self.atom_bond_block_list = nn.ModuleList()
        self.bond_angle_block_list = nn.ModuleList()

        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                BondEmbedding(self.bond_names, self.embed_dim))
            self.bond_float_rbf_list.append(
                BondFloatRBF(self.bond_float_names, self.embed_dim))
            self.bond_angle_float_rbf_list.append(
                BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim))
            self.atom_bond_block_list.append(
                GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1), batch_size=batch_size))
            self.bond_angle_block_list.append(
                GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1), batch_size=batch_size))
        
        self.graph_pool = global_mean_pool

    @profile
    def forward(self, graph_batch):
        """
        Build the network.
        Data is expected to be a torch_geometric.data.Data object containing
        - x: Node feature matrix
        - edge_index: Graph connectivity (COO format)
        - edge_attr: Edge feature matrix
        - batch: Batch vector mapping each node to its respective graph
        """
        node_hidden = self.init_atom_embedding(graph_batch)
        bond_embed = self.init_bond_embedding(graph_batch)
        edge_hidden = bond_embed + self.init_bond_float_rbf(graph_batch)

        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]
        for layer_id in range(self.layer_num):
            node_hidden = self.atom_bond_block_list[layer_id]( # time consuming 30%, 0.9s -> 50%, 1.3s
                node_hidden_list[layer_id],
                graph_batch['atom_bond_graph_edge'],
                edge_hidden_list[layer_id],
                graph_batch['atom_bond_graph_id'])
            
            cur_edge_hidden = self.bond_embedding_list[layer_id](graph_batch)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](graph_batch)
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](graph_batch)
            edge_hidden = self.bond_angle_block_list[layer_id]( # time consuming 55%, 1.8s -> 25%, 0.7s
                cur_edge_hidden,
                graph_batch['bond_angle_graph_edge'],
                cur_angle_hidden,
                graph_batch['bond_angle_graph_id'])
            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)
        
        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]
        graph_repr = self.graph_pool(node_repr, graph_batch['atom_bond_graph_id'])
        return node_repr, edge_repr, graph_repr

def calculate_molecule_meta_data_by_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    molecule_with_h = Chem.AddHs(molecule)
    AllChem.EmbedMultipleConfs(molecule_with_h, numConfs=10)
    # molecule_with_h.GetConformer 返回的是 molecule_with_h 的所有构象的列表
    try:
        result = AllChem.MMFFOptimizeMoleculeConfs(molecule_with_h)
    except:
        print('MMFFOptimizeMolecule error', smiles)
    molecule = Chem.RemoveHs(molecule_with_h)
    lowest_energy_conformation_index = int(np.argmin([x[1] for x in result]))
    coordinate_list = molecule.GetConformer(id=lowest_energy_conformation_index).GetPositions()
    coordinate_list = np.array(coordinate_list, dtype=np.float32)
    
    molecule_meta_data = {}
    molecule_meta_data['atom_pos'] = coordinate_list
    molecule_meta_data['edges'] = []
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        molecule_meta_data['edges'] += [(i, j), (j, i)]
    molecule_meta_data['edges'] += [(i, i) for i in range(molecule.GetNumAtoms())] # self loop

    atom_feature_name_list = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
    bond_feature_name_list = ["bond_dir", "bond_type", "is_in_ring"]
    atom_feature_name_to_possible_value_list_dict = {
        "atomic_num": list(range(1, 119)),
        "chiral_tag": list(Chem.ChiralType.values.values()),
        "degree": list(range(11)),
        "formal_charge": list(range(-5, 11)),
        "hybridization": list(Chem.HybridizationType.values.values()),
        "is_aromatic": [0, 1],
        "total_numHs": list(range(9)),
    }
    bond_feature_name_to_possible_value_list_dict = {
        "bond_dir": list(Chem.BondDir.values.values()),
        "bond_type": list(Chem.BondType.values.values()),
        "is_in_ring": [0, 1],
    }

    atom_feature_name_to_function_dict = {
        'atomic_num': lambda atom: atom.GetAtomicNum(),
        'chiral_tag': lambda atom: atom.GetChiralTag(),
        'degree': lambda atom: atom.GetDegree(),
        'formal_charge': lambda atom: atom.GetFormalCharge(),
        'hybridization': lambda atom: atom.GetHybridization(),
        'is_aromatic': lambda atom: int(atom.GetIsAromatic()),
        'total_numHs': lambda atom: atom.GetTotalNumHs(includeNeighbors=True),
    }
    bond_feature_name_to_function_dict = {
        'bond_dir': lambda bond: bond.GetBondDir(),
        'bond_type': lambda bond: bond.GetBondType(),
        'is_in_ring': lambda bond: int(bond.IsInRing()),
    }

    for atom_feature_name in atom_feature_name_list:
        molecule_meta_data[atom_feature_name] = []
        atom_feature_function = atom_feature_name_to_function_dict[atom_feature_name]
        atom_feature_possible_value_list = atom_feature_name_to_possible_value_list_dict[atom_feature_name]
        for atom in molecule.GetAtoms():
            atom_feature_value = atom_feature_function(atom)
            atom_feature_value_index = atom_feature_possible_value_list.index(atom_feature_value)
            molecule_meta_data[atom_feature_name].append(atom_feature_value_index + 1)

    for bond_feature_name in bond_feature_name_list:
        molecule_meta_data[bond_feature_name] = []
        bond_feature_function = bond_feature_name_to_function_dict[bond_feature_name]
        bond_feature_possible_value_list = bond_feature_name_to_possible_value_list_dict[bond_feature_name]
        for bond in molecule.GetBonds():
            bond_feature_value = bond_feature_function(bond)
            bond_feature_value_index = bond_feature_possible_value_list.index(bond_feature_value)
            molecule_meta_data[bond_feature_name] += [bond_feature_value_index + 1] * 2 # i->j and j->i
        self_loop_bond_feature_value_index = len(bond_feature_possible_value_list) + 2
        molecule_meta_data[bond_feature_name] += [self_loop_bond_feature_value_index] * molecule.GetNumAtoms() # self loop

    # for feature_name in atom_feature_name_list + bond_feature_name_list:
    #     molecule_meta_data[feature_name] = np.array(molecule_meta_data[feature_name], dtype=np.int64)
    # molecule_meta_data['edges'] = np.array(molecule_meta_data['edges'], dtype=np.int64)

    bond_length_list = []
    for source_atom_index, target_atom_index in molecule_meta_data['edges']:
        bond_length_list.append(np.linalg.norm(coordinate_list[target_atom_index] - coordinate_list[source_atom_index]))
    # molecule_meta_data['bond_length'] = np.array(bond_length_list, dtype=np.float32)

    def _calculate_bond_angle(vector_1, vector_2):
        vector_1_length = np.linalg.norm(vector_1)
        vector_2_length = np.linalg.norm(vector_2)
        if vector_1_length == 0 or vector_2_length == 0: # self loop
            return 0
        vector_1 = vector_1 / (vector_1_length + 1e-5)    # 1e-5: prevent numerical errors
        vector_2 = vector_2 / (vector_2_length + 1e-5)
        bond_angle = np.arccos(np.dot(vector_1, vector_2))
        return bond_angle

    edge_list = molecule_meta_data['edges']
    edge_pair_list = []
    bond_angle_list = []
    for edge_1_index, edge_1 in enumerate(edge_list):
        edge_2_index_list = []
        for edge_2_index, edge_2 in enumerate(edge_list):
            if edge_2[1] == edge_1[0] and edge_2_index != edge_1_index:
                edge_2_index_list.append(edge_2_index)
        
        for edge_2_index in edge_2_index_list:
            edge_2 = edge_list[edge_2_index]
            bond_vector_1 = coordinate_list[edge_1[1]] - coordinate_list[edge_1[0]]
            bond_vector_2 = coordinate_list[edge_2[1]] - coordinate_list[edge_2[0]]
            edge_pair_list.append([edge_2_index, edge_1_index])
            bond_angle = _calculate_bond_angle(bond_vector_2, bond_vector_1)
            bond_angle_list.append(bond_angle)

    molecule_meta_data['BondAngleGraph_edges'] = edge_pair_list
    molecule_meta_data['bond_angle'] = bond_angle_list

    # molecule_meta_data['BondAngleGraph_edges'] = np.array(edge_pair_list, dtype=np.int64)
    # molecule_meta_data['bond_angle'] = np.array(bond_angle_list, dtype=np.float32)
    return molecule_meta_data
    