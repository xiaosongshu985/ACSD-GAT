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

RDLogger.DisableLog('rdApp.*')

folder_path = '.'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def calculate_NumAromaticRings(smiles):
    values = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
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
        else:
            values.append([0.0])
    return values

def calculate_NumAromaticHeterocycles(smiles):
    values = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            n_aromatic_hetero = sum(1 for atom in mol.GetAtoms()
                                   if atom.GetIsAromatic() and atom.GetAtomicNum() not in [6, 1])
            values.append([float(n_aromatic_hetero)])
        else:
            values.append([0.0])
    return values

def calculate_MACCS_fingerprints(smiles):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fp_bits = [int(bit) for bit in fp.ToBitString()]
            fingerprints.append(fp_bits)
        else:
            fingerprints.append([0] * 167)
    return fingerprints

def calculate_morgan_fingerprints(smiles, nBits=2048, radius=2):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            fp_bits = list(fp)
            fingerprints.append(fp_bits)
        else:
            fingerprints.append([0] * nBits)
    return fingerprints

def calculate_RDK_fingerprints(smiles, nBits=2048):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=nBits)
            fp_bits = list(fp)
            fingerprints.append(fp_bits)
        else:
            fingerprints.append([0] * nBits)
    return fingerprints

def mol_to_graph(smiles, target, atom1_index, atom2_index):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        coords = torch.zeros((num_atoms, 3), dtype=torch.float32)
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            coords[i] = torch.tensor([pos.x, pos.y, pos.z])
        if torch.cuda.is_available():
            coords = coords.cuda()
        if torch.is_tensor(atom1_index):
            atom1_index = atom1_index.cpu().item() if atom1_index.is_cuda else atom1_index.item()
        if torch.is_tensor(atom2_index):
            atom2_index = atom2_index.cpu().item() if atom2_index.is_cuda else atom2_index.item()
        if atom1_index >= num_atoms or atom2_index >= num_atoms:
            raise ValueError(f"Atom index out of range: {atom1_index}, {atom2_index} (num_atoms: {num_atoms})")
        axis_center = (coords[atom1_index] + coords[atom2_index]) / 2.0
        diff = coords - axis_center.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
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
        if coords.is_cuda:
            node_features = node_features.cuda()
        edge_index = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append([start, end])
            edge_index.append([end, start])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.tensor([target], dtype=torch.float),
            axis_center=axis_center,
            distances=distances,
            axis_indices=torch.tensor([atom1_index, atom2_index], dtype=torch.long)
        )
        return data
    except Exception as e:
        raise RuntimeError(f"Error processing molecule {smiles}: {str(e)}")

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

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, heads=1, concat=True):
        super(GATLayer, self).__init__()
        self.conv = GATConv(in_features, out_features, heads=heads, dropout=dropout, concat=concat)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

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
        self.gat = GATEncoder(nfeat, out_features, nhid, dropout, nheads)
        self.fc1 = nn.Linear(3*out_dim + 2*out_features + 6, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.act_func = nn.ReLU()

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

    def _pool_local_gat_output(self, gat_out, mols, threshold=3.5):
        pooled_features = []
        node_offset = 0
        for mol in mols:
            num_nodes = mol.num_nodes
            mol_features = gat_out[node_offset:node_offset + num_nodes]
            local_mask = mol.distances <= threshold
            if local_mask.sum() > 0:
                local_pooled = torch.mean(mol_features[local_mask], dim=0)
            else:
                local_pooled = torch.mean(mol_features, dim=0)
            pooled_features.append(local_pooled)
            node_offset += num_nodes
        return torch.stack(pooled_features, dim=0)

    def forward(self, smiles, temperature, df_dc1, df_dc2, df_dc3, num_aromatic_heterocycles, num_aromatic_rings, atom1_index_list, atom2_index_list):
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
        mols = [mol_to_graph(smile, 0, atom1, atom2) for smile, atom1, atom2 in zip(smiles, atom1_index_list, atom2_index_list)]
        batched_x = torch.cat([mol.x for mol in mols], dim=0).to(device)
        batched_edge_index = self._build_batched_edge_index(mols)
        gat_out = self.gat(batched_x, batched_edge_index)
        global_pooled = self._pool_gat_output(gat_out, mols)
        local_pooled = self._pool_local_gat_output(gat_out, mols, threshold=3.5)
        gat_combined = torch.cat([global_pooled, local_pooled], dim=1)
        base_features = torch.cat([fpn_out_morgan, fpn_out_RDK, fpn_out_MACCS, gat_combined], dim=1)
        dc_features = torch.cat([df_dc1, df_dc2, df_dc3], dim=1)
        original_combined = torch.cat([base_features, temperature, dc_features,
                                     num_aromatic_heterocycles, num_aromatic_rings], dim=1)
        x = self.fc1(original_combined)
        return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, temperatures, df_dc1, df_dc2, df_dc3, targets, atom1_index_list, atom2_index_list):
        self.smiles = smiles
        self.temperatures = temperatures
        self.df_dc1 = df_dc1
        self.df_dc2 = df_dc2
        self.df_dc3 = df_dc3
        self.targets = targets
        self.atom1_index_list = atom1_index_list
        self.atom2_index_list = atom2_index_list
        self.num_aromatic_heterocycles = calculate_NumAromaticHeterocycles(smiles)
        self.num_aromatic_rings = calculate_NumAromaticRings(smiles)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return (
            self.smiles[idx],
            torch.tensor(self.temperatures[idx], dtype=torch.float),
            torch.tensor(self.df_dc1[idx], dtype=torch.float),
            torch.tensor(self.df_dc2[idx], dtype=torch.float),
            torch.tensor(self.df_dc3[idx], dtype=torch.float),
            torch.tensor(self.targets[idx], dtype=torch.float),
            torch.tensor(self.num_aromatic_heterocycles[idx], dtype=torch.float),
            torch.tensor(self.num_aromatic_rings[idx], dtype=torch.float),
            torch.tensor(self.atom1_index_list[idx], dtype=torch.long),
            torch.tensor(self.atom2_index_list[idx], dtype=torch.long)
        )

for file_name in csv_files:
    data = pd.read_csv(os.path.join(folder_path, file_name))
    smiles = data.iloc[:, 0].tolist()
    targets = data.iloc[:, 2].tolist()
    temperatures = data.iloc[:, 1].tolist()
    df_dc1 = data.iloc[:, 3].tolist()
    df_dc2 = data.iloc[:, 4].tolist()
    df_dc3 = data.iloc[:, 5].tolist()
    matrix_series = data.iloc[:, 7]
    atom1_index_list = matrix_series.str.strip('[]').str.split(',').str[0].str.strip().astype(int).tolist()
    atom2_index_list = matrix_series.str.strip('[]').str.split(',').str[1].str.strip().astype(int).tolist()

    (train_smiles, val_smiles, train_targets, val_targets,
     train_temperatures, val_temperatures, train_df_dc1, val_df_dc1,
     train_df_dc2, val_df_dc2, train_df_dc3, val_df_dc3,
     train_atom1_index, val_atom1_index, train_atom2_index, val_atom2_index) = train_test_split(
        smiles, targets, temperatures, df_dc1, df_dc2, df_dc3,
        atom1_index_list, atom2_index_list, test_size=0.1, random_state=42)
    batch_size = 32

    train_dataset = CustomDataset(
        train_smiles, train_temperatures, train_df_dc1, train_df_dc2, train_df_dc3, train_targets, train_atom1_index, train_atom2_index
    )
    val_dataset = CustomDataset(
        val_smiles, val_temperatures, val_df_dc1, val_df_dc2, val_df_dc3, val_targets, val_atom1_index, val_atom2_index
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    model = FpgnnModel(
        fp_dim_morgan=2048, fp_dim_RDK=2048, fp_dim_MACCS=167,
        hidden_dim=145, out_dim=11, dropout=0.3988525046296339, nfeat=6, nhid=105, nheads=11, out_features=21
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.002208599794914338)

    early_stopping_patience = 100
    min_epochs = 50
    best_val_r2 = -float('inf')
    best_epoch = 0
    best_model_path = f"ACSD-GAT-{file_name.split('.')[0]}-model.pth"

    for epoch in range(5000):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for batch in train_loader:
            (smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch,
             targets_batch, num_aromatic_heterocycles_batch, num_aromatic_rings_batch,
             atom1_index_list_batch, atom2_index_list_batch) = batch
            temp_batch = temp_batch.view(-1, 1).float().to(device)
            dc1_batch = dc1_batch.view(-1, 1).float().to(device)
            dc2_batch = dc2_batch.view(-1, 1).float().to(device)
            dc3_batch = dc3_batch.view(-1, 1).float().to(device)
            targets_batch = targets_batch.view(-1, 1).float().to(device)
            num_aromatic_heterocycles_batch = num_aromatic_heterocycles_batch.view(-1, 1).float().to(device)
            num_aromatic_rings_batch = num_aromatic_rings_batch.view(-1, 1).float().to(device)
            atom1_index_list_batch = atom1_index_list_batch.view(-1).long().to(device)
            atom2_index_list_batch = atom2_index_list_batch.view(-1).long().to(device)

            output = model(smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch, num_aromatic_heterocycles_batch, num_aromatic_rings_batch, atom1_index_list_batch, atom2_index_list_batch)
            loss = criterion(output, targets_batch)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                (smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch,
                 targets_batch, num_aromatic_heterocycles_batch, num_aromatic_rings_batch,
                 atom1_index_list_batch, atom2_index_list_batch) = batch
                temp_batch = temp_batch.view(-1, 1).float().to(device)
                dc1_batch = dc1_batch.view(-1, 1).float().to(device)
                dc2_batch = dc2_batch.view(-1, 1).float().to(device)
                dc3_batch = dc3_batch.view(-1, 1).float().to(device)
                targets_batch = targets_batch.view(-1, 1).float().to(device)
                num_aromatic_heterocycles_batch = num_aromatic_heterocycles_batch.view(-1, 1).float().to(device)
                num_aromatic_rings_batch = num_aromatic_rings_batch.view(-1, 1).float().to(device)
                atom1_index_list_batch = atom1_index_list_batch.view(-1).long().to(device)
                atom2_index_list_batch = atom2_index_list_batch.view(-1).long().to(device)

                output = model(smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch, num_aromatic_heterocycles_batch, num_aromatic_rings_batch, atom1_index_list_batch, atom2_index_list_batch)
                loss = criterion(output, targets_batch)
                val_loss += loss.item()

                val_preds.extend(output.cpu().numpy().flatten())
                val_targets.extend(targets_batch.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)

        val_r2 = r2_score(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

        train_preds = []
        train_targets = []
        with torch.no_grad():
            for batch in train_loader:
                (smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch,
                 targets_batch, num_aromatic_heterocycles_batch, num_aromatic_rings_batch,
                 atom1_index_list_batch, atom2_index_list_batch) = batch
                temp_batch = temp_batch.view(-1, 1).float().to(device)
                dc1_batch = dc1_batch.view(-1, 1).float().to(device)
                dc2_batch = dc2_batch.view(-1, 1).float().to(device)
                dc3_batch = dc3_batch.view(-1, 1).float().to(device)
                targets_batch = targets_batch.view(-1, 1).float().to(device)
                num_aromatic_heterocycles_batch = num_aromatic_heterocycles_batch.view(-1, 1).float().to(device)
                num_aromatic_rings_batch = num_aromatic_rings_batch.view(-1, 1).float().to(device)
                atom1_index_list_batch = atom1_index_list_batch.view(-1).long().to(device)
                atom2_index_list_batch = atom2_index_list_batch.view(-1).long().to(device)

                output = model(smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch, num_aromatic_heterocycles_batch, num_aromatic_rings_batch, atom1_index_list_batch, atom2_index_list_batch)
                train_preds.extend(output.cpu().numpy().flatten())
                train_targets.extend(targets_batch.cpu().numpy().flatten())

        train_r2 = r2_score(train_targets, train_preds)
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}, '
              f'Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}')

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        if epoch - best_epoch >= early_stopping_patience and epoch >= min_epochs:
            print(f"Early stopping triggered after epoch {epoch + 1}. Best validation loss: {avg_val_loss:.4f}")
            break

    print(f"Training completed. Best validation loss: {avg_val_loss:.4f} at epoch {best_epoch + 1}")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    with torch.no_grad():
        val_preds = []
        val_targets = []
        for batch in val_loader:
            (smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch,
             targets_batch, num_aromatic_heterocycles_batch, num_aromatic_rings_batch,
             atom1_index_list_batch, atom2_index_list_batch) = batch
            temp_batch = temp_batch.view(-1, 1).float().to(device)
            dc1_batch = dc1_batch.view(-1, 1).float().to(device)
            dc2_batch = dc2_batch.view(-1, 1).float().to(device)
            dc3_batch = dc3_batch.view(-1, 1).float().to(device)
            targets_batch = targets_batch.view(-1, 1).float().to(device)
            num_aromatic_heterocycles_batch = num_aromatic_heterocycles_batch.view(-1, 1).float().to(device)
            num_aromatic_rings_batch = num_aromatic_rings_batch.view(-1, 1).float().to(device)
            atom1_index_list_batch = atom1_index_list_batch.view(-1).long().to(device)
            atom2_index_list_batch = atom2_index_list_batch.view(-1).long().to(device)

            output = model(smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch, num_aromatic_heterocycles_batch, num_aromatic_rings_batch, atom1_index_list_batch, atom2_index_list_batch)
            val_preds.extend(output.cpu().numpy().flatten())
            val_targets.extend(targets_batch.cpu().numpy().flatten())

        val_r2 = r2_score(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

        train_preds = []
        train_targets = []
        for batch in train_loader:
            (smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch,
             targets_batch, num_aromatic_heterocycles_batch, num_aromatic_rings_batch,
             atom1_index_list_batch, atom2_index_list_batch) = batch
            temp_batch = temp_batch.view(-1, 1).float().to(device)
            dc1_batch = dc1_batch.view(-1, 1).float().to(device)
            dc2_batch = dc2_batch.view(-1, 1).float().to(device)
            dc3_batch = dc3_batch.view(-1, 1).float().to(device)
            targets_batch = targets_batch.view(-1, 1).float().to(device)
            num_aromatic_heterocycles_batch = num_aromatic_heterocycles_batch.view(-1, 1).float().to(device)
            num_aromatic_rings_batch = num_aromatic_rings_batch.view(-1, 1).float().to(device)
            atom1_index_list_batch = atom1_index_list_batch.view(-1).long().to(device)
            atom2_index_list_batch = atom2_index_list_batch.view(-1).long().to(device)

            output = model(smiles_batch, temp_batch, dc1_batch, dc2_batch, dc3_batch, num_aromatic_heterocycles_batch, num_aromatic_rings_batch, atom1_index_list_batch, atom2_index_list_batch)
            train_preds.extend(output.cpu().numpy().flatten())
            train_targets.extend(targets_batch.cpu().numpy().flatten())

        train_r2 = r2_score(train_targets, train_preds)
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))

    print(f"Final Validation Metrics (Best Model): "
          f"Loss: {avg_val_loss:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    print(f"Final Training Metrics (Best Model): "
          f"RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")

    results_name = f"45-GAT-Morgan_RDK_MACCS-{file_name.split('.')[0]}-results.txt"
    results = {
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'best_val_loss': avg_val_loss,
        'best_epoch': best_epoch + 1
    }

    with open(results_name, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(f"Final Validation Metrics (Best Model): "
          f"Loss: {avg_val_loss:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    print(f"Final Training Metrics (Best Model): "
          f"RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")

    plt.figure(figsize=(8, 8), frameon=False)

    plt.scatter(train_targets, train_preds, alpha=0.3, edgecolor='black', facecolor='blue', label='Training Set')
    plt.scatter(val_targets, val_preds, alpha=0.6, edgecolor='black', facecolor='lightgreen', label='Validation Set')

    model = LinearRegression(fit_intercept=False)
    model.fit(
        np.concatenate([train_targets, val_targets]).reshape(-1, 1),
        np.concatenate([train_preds, val_preds])
    )

    slope = model.coef_[0]
    intercept = model.intercept_

    x_min = min(min(train_targets), min(val_targets))
    x_max = max(max(train_targets), max(val_targets))
    x_values = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_line = slope * x_values + intercept
    plt.plot(x_values, y_line, "r-", label=f'R² = {model.score(np.concatenate([train_targets, val_targets]).reshape(-1, 1), np.concatenate([train_preds, val_preds])):.2f}')

    plt.xlim([x_min, x_max])
    plt.ylim([x_min, x_max])

    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'45-GAT-Morgan_RDK_MACCS Prediction for {file_name}', fontsize=16)

    plt.legend().remove()

    plt.text(0.95, 0.15, f'Train RMSE: {train_rmse:.2f}\nR²: {train_r2:.2f}',
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right',
             fontdict=dict(color='blue', weight='bold'))

    plt.text(0.95, 0.05, f'Test RMSE: {val_rmse:.2f}\nR²: {val_r2:.2f}',
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right',
             fontdict=dict(color='green', weight='bold'))

    plt.grid(False)

    image_name = f"45-GAT-Morgan_RDK_MACCS-{file_name.split('.')[0]}.png"
    plt.savefig(image_name, transparent=True, bbox_inches='tight')
    plt.close()
