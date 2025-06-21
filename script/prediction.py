import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from torch_geometric.nn import GATConv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from rdkit import RDLogger
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys, Descriptors

# 禁用 RDKit 日志输出
RDLogger.DisableLog('rdApp.*')

# Set the working directory to the folder path
folder_path = '.'  # Set your folder path here

# Get all CSV files in the current directory
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Check if CUDA is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def calculate_NumAromaticRings(smiles):
    values = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            # 获取分子中的所有环
            sssr = Chem.rdmolops.GetSSSR(mol)
            num_aromatic_rings = 0
            for ring in sssr:
                # 检查环是否为芳香环
                is_aromatic = True
                for idx in ring:
                    if not mol.GetAtomWithIdx(idx).GetIsAromatic():
                        is_aromatic = False
                        break
                if is_aromatic:
                    num_aromatic_rings += 1
            values.append([float(num_aromatic_rings)])
        else:
            values.append([0.0])
    return values

# 添加计算NumAromaticHeterocycles的函数
def calculate_NumAromaticHeterocycles(smiles):
    values = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            # 计算芳香环杂原子数量
            n_aromatic_hetero = sum(1 for atom in mol.GetAtoms() 
                                   if atom.GetIsAromatic() and atom.GetAtomicNum() not in [6, 1])
            values.append([float(n_aromatic_hetero)])
        else:
            values.append([0.0])
    return values

# Define a function to calculate MACCS fingerprints
def calculate_MACCS_fingerprints(smiles):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)  # Generate MACCS fingerprint
            fp_bits = [int(bit) for bit in fp.ToBitString()]  # Convert fingerprint to bit string and then to list
            fingerprints.append(fp_bits)
        else:
            fingerprints.append([0] * 167)  # Fill with zeros if the molecule cannot be loaded
    return fingerprints

# Define a function to calculate Morgan fingerprints
def calculate_morgan_fingerprints(smiles, nBits=2048, radius=2):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            fp_bits = list(fp)
            fingerprints.append(fp_bits)
        else:
            fingerprints.append([0] * nBits)  # Fill with zeros if the molecule cannot be loaded
    return fingerprints

# Define a function to calculate RDK fingerprints
def calculate_RDK_fingerprints(smiles, nBits=2048):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=nBits)
            fp_bits = list(fp)
            fingerprints.append(fp_bits)
        else:
            fingerprints.append([0] * nBits)  # Fill with zeros if the molecule cannot be loaded
            print(f"{s}")
    return fingerprints



# Define a function to convert SMILES to graph data
def mol_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),  # Atomic number
            atom.GetDegree(),  # Degree
            atom.GetFormalCharge(),  # Formal charge
            int(atom.GetChiralTag()),  # Chirality
            atom.GetTotalNumHs(),  # Hydrogen count
            int(atom.GetHybridization())  # Hybridization state
        ])
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Edge features
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])  # Add reverse edge
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create PyG data object
    data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([target], dtype=torch.float))
    return data

# Define the FPN model
class FPN(nn.Module):
    def __init__(self, fp_dim, hidden_dim, out_dim, dropout):
        super(FPN, self).__init__()
        self.fc1 = nn.Linear(fp_dim, hidden_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, fp_list):
        fp_list = torch.tensor(fp_list, dtype=torch.float).to(device)
        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out

# Define the GATLayer using GATConv
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, heads=1, concat=True):
        super(GATLayer, self).__init__()
        self.conv = GATConv(in_features, out_features, heads=heads, dropout=dropout, concat=concat)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# Define the GATEncoder
class GATEncoder(nn.Module):
    def __init__(self, nfeat, out_features, nhid, dropout, nheads):
        super(GATEncoder, self).__init__()
        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout, heads=1, concat=True) for _ in range(nheads)
        ])
        self.out_att = GATLayer(nhid * nheads, out_features, dropout, heads=1, concat=False)

    def forward(self, atom_features, edge_index):
        atom_features = torch.cat([att(atom_features, edge_index) for att in self.attentions], dim=1)
        atom_features = self.out_att(atom_features, edge_index)
        return atom_features

# Define the FpgnnModel with combined fingerprints and three fully connected layers
class FpgnnModel(nn.Module):
    def __init__(self, fp_dim_morgan, fp_dim_RDK, fp_dim_MACCS, hidden_dim, out_dim, dropout, nfeat, nhid, nheads, out_features):
        super(FpgnnModel, self).__init__()
        self.fpn_morgan = FPN(fp_dim_morgan, hidden_dim, out_dim, dropout)
        self.fpn_RDK = FPN(fp_dim_RDK, hidden_dim, out_dim, dropout)
        self.fpn_MACCS = FPN(fp_dim_MACCS, hidden_dim, out_dim, dropout)
        self.gat = GATEncoder(nfeat,out_features, nhid, dropout, nheads)
        self.fc1 = nn.Linear(3*out_dim + out_features + 6, 1)
        self.dropout = nn.Dropout(p=dropout)
        #self.act_func = nn.ReLU()
        

    def _build_batched_edge_index(self, mols):
        offset = 0
        batched_edge_index = []
        for mol in mols:
            edge_index = mol.edge_index + offset
            batched_edge_index.append(edge_index)
            offset += mol.num_nodes
        batched_edge_index = torch.cat(batched_edge_index, dim=1).to(device)
        return batched_edge_index
    
    def _pool_gat_output(self, gat_out, mols):
        pooled_features = []
        offset = 0
        for mol in mols:
            num_nodes = mol.num_nodes
            mol_features = gat_out[offset:offset + num_nodes]
            pooled_feature = torch.mean(mol_features, dim=0)
            pooled_features.append(pooled_feature)
            offset += num_nodes
        return torch.stack(pooled_features, dim=0)

    def forward(self, smiles, temperature, df_dc1, df_dc2, df_dc3, num_aromatic_heterocycles, num_aromatic_rings):
        fp_morgan = calculate_morgan_fingerprints(smiles)
        fp_RDK = calculate_RDK_fingerprints(smiles)
        fp_MACCS = calculate_MACCS_fingerprints(smiles)
        
        num_aromatic_heterocycles = calculate_NumAromaticHeterocycles(smiles)
        num_aromatic_rings = calculate_NumAromaticRings(smiles)
        
        num_aromatic_heterocycles = torch.tensor(num_aromatic_heterocycles, dtype=torch.float).to(device)
        num_aromatic_rings = torch.tensor(num_aromatic_rings, dtype=torch.float).to(device)
        
        fpn_out_morgan = self.fpn_morgan(fp_morgan)
        fpn_out_RDK = self.fpn_RDK(fp_RDK)
        fpn_out_MACCS = self.fpn_MACCS(fp_MACCS)
            
        mols = [mol_to_graph(smile, 0) for smile in smiles]
        batched_x = torch.cat([mol.x for mol in mols], dim=0).to(device)
        batched_edge_index = self._build_batched_edge_index(mols)
        gat_out = self.gat(batched_x, batched_edge_index)
        gat_out = self._pool_gat_output(gat_out, mols)
            
        base_features = torch.cat([fpn_out_morgan, fpn_out_RDK, fpn_out_MACCS, gat_out], dim=1)
        dc_features = torch.cat([df_dc1, df_dc2, df_dc3], dim=1)
            
        # 将NumAromaticHeterocycles和NumAromaticRings加入到全连接层的输入中
        original_combined = torch.cat([base_features, temperature, dc_features, num_aromatic_heterocycles, num_aromatic_rings], dim=1)
        x = self.fc1(original_combined)
            
        return x
# FPN, GATLayer, GATEncoder, and FpgnnModel class definitions remain unchanged

# Load the trained model
model = FpgnnModel(
    fp_dim_morgan=2048, fp_dim_RDK=2048, fp_dim_MACCS=167,
        hidden_dim=145, out_dim=11,  dropout=0.3988525046296339, nfeat=6, nhid=105, nheads=11, out_features=21
).to(device)

best_model_path = "/home/xhs/work/3/best/opt/BEST/realbeat/45-GAT-Morgan_RDK_MACCS-all_exp+mtkl4-model.pth"
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Set the working directory and get all CSV files
folder_path = '.'  
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file_name in csv_files:
    data = pd.read_csv(os.path.join(folder_path, file_name))
    smiles = data.iloc[:, 0].tolist()
    #targets = data.iloc[:, 3].tolist()  # Assuming this is your actual value
    temperatures = data.iloc[:, 1].tolist()  
    df_dc1 = data.iloc[:, 2].tolist()
    df_dc2 = data.iloc[:, 3].tolist()
    df_dc3 = data.iloc[:, 4].tolist()

    temperature_tensor = torch.tensor(temperatures, dtype=torch.float).view(-1, 1).to(device)
    df_dc1_tensor = torch.tensor(df_dc1, dtype=torch.float).view(-1, 1).to(device)
    df_dc2_tensor = torch.tensor(df_dc2, dtype=torch.float).view(-1, 1).to(device)
    df_dc3_tensor = torch.tensor(df_dc3, dtype=torch.float).view(-1, 1).to(device)
    num_aromatic_heterocycles = calculate_NumAromaticHeterocycles(smiles)
    num_aromatic_rings = calculate_NumAromaticRings(smiles)
    num_aromatic_heterocycles_tensor = torch.tensor(num_aromatic_heterocycles, dtype=torch.float).to(device)
    num_aromatic_rings_tensor = torch.tensor(num_aromatic_rings, dtype=torch.float).to(device)

    with torch.no_grad():
        predictions = model(smiles, temperature_tensor, df_dc1_tensor, df_dc2_tensor, df_dc3_tensor, num_aromatic_heterocycles_tensor, num_aromatic_rings_tensor)

    predicted_values = predictions.cpu().numpy().flatten()
    for smile, prediction in zip(smiles, predicted_values):
        print(f"SMILES: {smile}, Predicted Value: {prediction:.4f}")