import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from collections import defaultdict
import os

def cluster_molecules_by_scaffold(smiles_list, n_clusters=20):
    """
    Cluster molecules by Murcko-scaffold similarity.
    """
    print("Step 1: Extract Murcko scaffolds and generate fingerprints...")
    valid_data = []
    for idx, smi in enumerate(smiles_list):
        if pd.isna(smi) or smi == '':
            continue
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold_mol is None:
                continue
            scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(scaffold_mol, radius=2, nBits=1024)
            valid_data.append({
                'original_smiles': smi,
                'scaffold_mol': scaffold_mol,
                'fingerprint': scaffold_fp
            })
        except Exception as e:
            print(f"Error processing SMILES '{smi}': {e}")
            continue
    print(f"Successfully processed {len(valid_data)} valid molecules")
    if len(valid_data) < n_clusters:
        n_clusters = len(valid_data)
    if len(valid_data) == 0:
        return {}
    print("Step 2: Compute distance matrix...")
    fps = [item['fingerprint'] for item in valid_data]
    nfps = len(fps)
    dists = []
    for i in range(1, nfps):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in similarities])
    print("Step 3: Perform Butina clustering...")
    cutoff = 0.3
    cluster_indices = list(Butina.ClusterData(dists, nfps, cutoff, True))
    cluster_indices.sort(key=len, reverse=True)
    print(f"Initial clustering yielded {len(cluster_indices)} clusters")
    scaffold_clusters = defaultdict(list)
    if len(cluster_indices) > n_clusters:
        main_clusters = cluster_indices[:n_clusters]
        for cluster_idx, cluster in enumerate(main_clusters):
            for mol_idx in cluster:
                scaffold_clusters[cluster_idx].append(valid_data[mol_idx]['original_smiles'])
        remaining_indices = []
        for cluster in cluster_indices[n_clusters:]:
            remaining_indices.extend(cluster)
        if remaining_indices:
            main_cluster_fps = [fps[cluster[0]] for cluster in main_clusters if cluster]
            for mol_idx in remaining_indices:
                mol_fp = fps[mol_idx]
                similarities = [DataStructs.TanimotoSimilarity(mol_fp, mc_fp) for mc_fp in main_cluster_fps]
                best_cluster = np.argmax(similarities) if similarities else 0
                scaffold_clusters[best_cluster].append(valid_data[mol_idx]['original_smiles'])
    else:
        for cluster_idx, cluster in enumerate(cluster_indices):
            for mol_idx in cluster:
                scaffold_clusters[cluster_idx].append(valid_data[mol_idx]['original_smiles'])
    actual_clusters = len(scaffold_clusters)
    if actual_clusters > n_clusters:
        cluster_items = sorted(scaffold_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        scaffold_clusters = defaultdict(list)
        for i, (_, mols) in enumerate(cluster_items[:n_clusters]):
            scaffold_clusters[i] = mols
    print(f"Final number of clusters: {len(scaffold_clusters)}")
    return dict(scaffold_clusters)

def cluster_and_label_data(csv_file_path, smiles_column_index=1, n_clusters=20, output_file="clustered_molecules.csv"):
    """
    Main function: read CSV, cluster by scaffold, export labeled CSV.
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read file: {csv_file_path}")
        print(f"Data shape: {df.shape}")
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None

    if smiles_column_index >= len(df.columns):
        print(f"Error: file has only {len(df.columns)} columns, but column index {smiles_column_index+1} was specified")
        return None

    smiles_col_name = df.columns[smiles_column_index]
    smiles_list = df[smiles_col_name].tolist()
    print(f"SMILES column name: '{smiles_col_name}'")
    print(f"Total molecules: {len(smiles_list)}")

    clustered_smiles = cluster_molecules_by_scaffold(smiles_list, n_clusters)
    if not clustered_smiles:
        print("Clustering failed, cannot generate labeled file")
        return None

    print("\nCluster statistics:")
    for cluster_id, molecules in clustered_smiles.items():
        print(f"Cluster {cluster_id}: {len(molecules)} molecules")

    smiles_to_cluster = {}
    for cluster_id, cluster_smiles in clustered_smiles.items():
        for smi in cluster_smiles:
            smiles_to_cluster[str(smi)] = cluster_id

    cluster_labels = []
    for smi in smiles_list:
        if pd.isna(smi) or smi == '':
            cluster_labels.append(-1)  # invalid SMILES
        elif str(smi) in smiles_to_cluster:
            cluster_labels.append(smiles_to_cluster[str(smi)])
        else:
            cluster_labels.append(-1)  # unclustered

    df_clustered = df.copy()
    if smiles_column_index + 1 < len(df.columns):
        df_clustered.insert(smiles_column_index + 1, 'cluster_id', cluster_labels)
    else:
        df_clustered['cluster_id'] = cluster_labels

    cluster_counts = df_clustered['cluster_id'].value_counts().sort_index()
    print("\nCluster distribution:")
    for cluster_id, count in cluster_counts.items():
        if cluster_id == -1:
            print(f"  Unclustered: {count} molecules")
        else:
            print(f"  Cluster {cluster_id}: {count} molecules")

    try:
        df_clustered.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved clustered data to: {output_file}")
        print(f"File contains {len(df_clustered)} molecules, {len(clustered_smiles)} clusters")
        print(f"\nFirst 5 rows preview:")
        print(df_clustered.head())
    except Exception as e:
        print(f"Failed to save file: {e}")
        return None

    return df_clustered

# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    input_file       = "all_exp+mtkl-zz.csv"  # replace with your CSV
    smiles_col_index = 1                      # SMILES in 2nd column (0-based)
    n_clusters       = 20
    output_file      = "clustered_molecules.csv"

    clustered_data = cluster_and_label_data(
        csv_file_path=input_file,
        smiles_column_index=smiles_col_index,
        n_clusters=n_clusters,
        output_file=output_file
    )

    if clustered_data is not None:
        print("\nClustering complete!")
        print(f"Output file: {output_file}")
        print(f"Total molecules: {len(clustered_data)}")
        n_clu = clustered_data['cluster_id'].n_unique() - (1 if -1 in clustered_data['cluster_id'].values else 0)
        print(f"Number of clusters: {n_clu}")

        clustered_only = clustered_data[clustered_data['cluster_id'] != -1]
        if len(clustered_only) > 0:
            avg_size = len(clustered_only) / clustered_only['cluster_id'].nunique()
            print(f"Average cluster size: {avg_size:.1f} molecules")
    else:
        print("Clustering failed!")
