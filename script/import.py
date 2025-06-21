import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import shap
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt

# 定义计算特征的函数
def calculate_NumAromaticRings(smiles):
    values = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        sssr = Chem.rdmolops.GetSSSR(mol)
        num_aromatic_rings = 0
        for ring in sssr:
            is_aromatic = True
            for idx in ring:
                if not mol.GetAtomWithIdx(idx).GetIsAromatic():
                    is_aromatic = False
                    break
            if is_aromatic:
                num_aromatic_rings += 1
        values.append([float(num_aromatic_rings)])
    return values

def calculate_NumAromaticHeterocycles(smiles):
    values = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        n_aromatic_hetero = sum(1 for atom in mol.GetAtoms() 
                               if atom.GetIsAromatic() and atom.GetAtomicNum() not in [6, 1])
        values.append([float(n_aromatic_hetero)])
    return values

def calculate_MACCS_fingerprints(smiles):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_bits = [int(bit) for bit in fp.ToBitString()]
        fingerprints.append(fp_bits)
    return fingerprints

def calculate_morgan_fingerprints(smiles, nBits=2048, radius=2):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        fp_bits = list(fp)
        fingerprints.append(fp_bits)
    return fingerprints

def calculate_RDK_fingerprints(smiles, nBits=2048):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=nBits)
        fp_bits = list(fp)
        fingerprints.append(fp_bits)
    return fingerprints

def mol_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetChiralTag()),
            atom.GetTotalNumHs(),
            int(atom.GetHybridization())
        ])
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([target], dtype=torch.float))
    data.smiles = smiles
    data.temp = torch.tensor([0.0])
    data.dc1 = torch.tensor([0.0])
    data.dc2 = torch.tensor([0.0])
    data.dc3 = torch.tensor([0.0])
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

    def forward(self, smiles, temperature, df_dc1, df_dc2, df_dc3):
        combined = self.get_combined_features(smiles, temperature, df_dc1, df_dc2, df_dc3)
        return self.fc1(combined)

    def get_combined_features(self, smiles, temperature, df_dc1, df_dc2, df_dc3):
        # 确保所有输入张量都在设备上
        temperature = temperature.to(device)
        df_dc1 = df_dc1.to(device)
        df_dc2 = df_dc2.to(device)
        df_dc3 = df_dc3.to(device)
        
        fp_morgan = self.fpn_morgan(calculate_morgan_fingerprints(smiles)).to(device)
        fp_rdk = self.fpn_RDK(calculate_RDK_fingerprints(smiles)).to(device)
        fp_maccs = self.fpn_MACCS(calculate_MACCS_fingerprints(smiles)).to(device)
        
        mols = [mol_to_graph(s, 0) for s in smiles]
        batched_x = torch.cat([mol.x for mol in mols], dim=0).to(device)
        edge_index = self._build_batched_edge_index(mols).to(device)
        gat_out = self.gat(batched_x, edge_index)
        gat_pooled = self._pool_gat_output(gat_out, mols)
        
        aromatic_ft = torch.cat([
            torch.tensor(calculate_NumAromaticHeterocycles(smiles), dtype=torch.float).to(device),
            torch.tensor(calculate_NumAromaticRings(smiles), dtype=torch.float).to(device)
        ], dim=1).to(device)
        
        combined = torch.cat([
            fp_morgan, fp_rdk, fp_maccs, gat_pooled,
            temperature, df_dc1, df_dc2, df_dc3,
            aromatic_ft
        ], dim=1)
        
        return combined

    def predict_for_shap(self, x):
        return self.fc1(x)

# 加载数据
file_path = '/home/xhs/work/3/best/opt/BEST/realbeat/all_exp+mtkl4.csv'
data = pd.read_csv(file_path)
smiles = data.iloc[:, 1].tolist()
targets = data.iloc[:, 3].tolist()
temperatures = data.iloc[:, 2].tolist()
df_dc1 = data.iloc[:, 4].tolist()
df_dc2 = data.iloc[:, 5].tolist()
df_dc3 = data.iloc[:, 6].tolist()

# 不进行数据划分，直接使用整个数据集
all_data = []
for i in range(len(smiles)):
    data_point = mol_to_graph(smiles[i], targets[i])
    data_point.temp = torch.tensor([temperatures[i]], dtype=torch.float).view(-1)
    data_point.dc1 = torch.tensor([df_dc1[i]], dtype=torch.float).view(-1)
    data_point.dc2 = torch.tensor([df_dc2[i]], dtype=torch.float).view(-1)
    data_point.dc3 = torch.tensor([df_dc3[i]], dtype=torch.float).view(-1)
    all_data.append(data_point)



all_loader = DataLoader(all_data, batch_size=32, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FpgnnModel(
        fp_dim_morgan=2048, fp_dim_RDK=2048, fp_dim_MACCS=167,
        hidden_dim=145, out_dim=11,  dropout=0.3988525046296339, nfeat=6, nhid=105, nheads=11, out_features=21
    ).to(device)

model_path = '/home/xhs/work/3/best/opt/BEST/realbeat/45-GAT-Morgan_RDK_MACCS-all_exp+mtkl4-model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def shap_analysis_batch(model, test_data, feature_names, feature_blocks):
    import numpy as np
    import matplotlib.pyplot as plt
    import shap

    n_samples = min(1015, len(test_data))
    test_samples = test_data[:n_samples]
    
    # 确保所有张量都在设备上
    smiles_batch = [d.smiles for d in test_samples]
    temp_batch = torch.stack([d.temp for d in test_samples]).to(device)
    dc1_batch = torch.stack([d.dc1 for d in test_samples]).to(device)
    dc2_batch = torch.stack([d.dc2 for d in test_samples]).to(device)
    dc3_batch = torch.stack([d.dc3 for d in test_samples]).to(device)
    
    test_features = model.get_combined_features(
        smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch
    ).detach().cpu().numpy()

    def predict_fn(x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            return model.predict_for_shap(x_tensor).detach().cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, test_features[:1015])
    shap_values = explainer.shap_values(test_features)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])

    # 计算各特征分段长度
    morgan_len = feature_blocks[0]
    rdk_len = feature_blocks[1]
    maccs_len = feature_blocks[2]
    gat_len = feature_blocks[3]

    morgan_start = 0
    morgan_end = morgan_start + morgan_len
    rdk_start = morgan_end
    rdk_end = rdk_start + rdk_len
    maccs_start = rdk_end
    maccs_end = maccs_start + maccs_len
    gat_start = maccs_end
    gat_end = gat_start + gat_len

    # 合并各特征为sum
    morgan_sum_feat = test_features[:, morgan_start:morgan_end].sum(axis=1, keepdims=True)
    morgan_sum_shap = shap_values[:, morgan_start:morgan_end].sum(axis=1, keepdims=True)
    rdk_sum_feat = test_features[:, rdk_start:rdk_end].sum(axis=1, keepdims=True)
    rdk_sum_shap = shap_values[:, rdk_start:rdk_end].sum(axis=1, keepdims=True)
    maccs_sum_feat = test_features[:, maccs_start:maccs_end].sum(axis=1, keepdims=True)
    maccs_sum_shap = shap_values[:, maccs_start:maccs_end].sum(axis=1, keepdims=True)
    gat_sum_feat = test_features[:, gat_start:gat_end].sum(axis=1, keepdims=True)
    gat_sum_shap = shap_values[:, gat_start:gat_end].sum(axis=1, keepdims=True)

    # 拼接新特征和shap值
    test_features_new = np.concatenate([
        morgan_sum_feat,
        rdk_sum_feat,
        maccs_sum_feat,
        gat_sum_feat,
        test_features[:, gat_end:]  # 其余特征（温度、dc1、dc2、dc3、芳香环等）
    ], axis=1)
    shap_values_new = np.concatenate([
        morgan_sum_shap,
        rdk_sum_shap,
        maccs_sum_shap,
        gat_sum_shap,
        shap_values[:, gat_end:]
    ], axis=1)

    # 新特征名
    feature_names_new = (
        ["Morgan_sum", "RDK_sum", "MACCS_sum", "GAT_sum"] +
        feature_names[gat_end:]
    )

    # 去掉温度
    if "Morgan_sum" in feature_names_new:
        Morgan_sum_idx = feature_names_new.index("Morgan_sum")
        feature_names_new.pop(Morgan_sum_idx)
        test_features_new = np.delete(test_features_new, Morgan_sum_idx, axis=1)
        shap_values_new = np.delete(shap_values_new, Morgan_sum_idx, axis=1)
    
    # 去掉温度
    if "RDK_sum" in feature_names_new:
        RDK_sum_idx = feature_names_new.index("RDK_sum")
        feature_names_new.pop(RDK_sum_idx)
        test_features_new = np.delete(test_features_new, RDK_sum_idx, axis=1)
        shap_values_new = np.delete(shap_values_new, RDK_sum_idx, axis=1)
    
    if "MACCS_sum" in feature_names_new:
        MACCS_sum_idx = feature_names_new.index("MACCS_sum")
        feature_names_new.pop(MACCS_sum_idx)
        test_features_new = np.delete(test_features_new, MACCS_sum_idx, axis=1)
        shap_values_new = np.delete(shap_values_new, MACCS_sum_idx, axis=1)
    
    if "GAT_sum" in feature_names_new:
        GAT_sum_idx = feature_names_new.index("GAT_sum")
        feature_names_new.pop(GAT_sum_idx)
        test_features_new = np.delete(test_features_new, GAT_sum_idx, axis=1)
        shap_values_new = np.delete(shap_values_new, GAT_sum_idx, axis=1)

    # 检查长度
    assert test_features_new.shape[1] == shap_values_new.shape[1] == len(feature_names_new), \
        f"维度不一致: {test_features_new.shape[1]}, {shap_values_new.shape[1]}, {len(feature_names_new)}"

    plt.figure(figsize=(15, 8))
    shap.summary_plot(shap_values_new, test_features_new, feature_names=feature_names_new, show=False)
    plt.savefig("global_feature_importance.png")
    print("SHAP分析完成，结果已保存为global_feature_importance.png。")

feature_blocks = [
    model.fpn_morgan.fc2.out_features,
    model.fpn_RDK.fc2.out_features,
    model.fpn_MACCS.fc2.out_features,
    model.gat.out_att.conv.out_channels,
    4,
    2
]
feature_names = (
    [f"Morgan_{i}" for i in range(feature_blocks[0])] +
    [f"RDK_{i}" for i in range(feature_blocks[1])] +
    [f"MACCS_{i}" for i in range(feature_blocks[2])] +
    [f"GAT_{i}" for i in range(feature_blocks[3])] +
    ["Temperature", "overlapping_volume1", "overlapping_volume2", "Buried_volume"] +
    ["AromaticHetero", "AromaticRings"]
)

shap_analysis_batch(model, all_data, feature_names, feature_blocks)