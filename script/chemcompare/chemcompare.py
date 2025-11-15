import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.Chem.Draw import MolsToGridImage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(train_path, test_path, smiles_col='SMILES'):
    """
    Load and preprocess train and test datasets
    """
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Convert SMILES to molecules
    train_df['mol'] = train_df[smiles_col].apply(Chem.MolFromSmiles)
    test_df['mol'] = test_df[smiles_col].apply(Chem.MolFromSmiles)

    # Remove invalid molecules
    train_df = train_df[train_df['mol'].notnull()].copy()
    test_df = test_df[test_df['mol'].notnull()].copy()

    print(f"Training set valid molecules: {len(train_df)}")
    print(f"Test set valid molecules: {len(test_df)}")

    return train_df, test_df


def compute_similarity_metrics(train_df, test_df):
    """
    Compute various similarity metrics between datasets
    """
    # Generate Morgan fingerprints
    train_df['fingerprint'] = train_df['mol'].apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) if x else None
    )
    test_df['fingerprint'] = test_df['mol'].apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) if x else None
    )

    # Remove molecules without fingerprints
    train_fps = [fp for fp in train_df['fingerprint'] if fp is not None]
    test_fps = [fp for fp in test_df['fingerprint'] if fp is not None]

    # Calculate intra-set similarities
    def intra_set_similarity(fingerprints):
        if len(fingerprints) < 2:
            return 0.0
        similarities = []
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):  # Sample for efficiency
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarities.append(sim)
        return np.mean(similarities) if similarities else 0.0

    train_intra_sim = intra_set_similarity(train_fps)
    test_intra_sim = intra_set_similarity(test_fps)

    # Calculate inter-set similarity
    inter_similarities = []
    for test_fp in test_fps:  # Sample for efficiency
        sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        inter_similarities.extend(sims)

    inter_sim = np.mean(inter_similarities) if inter_similarities else 0.0

    return {
        'train_intra_similarity': train_intra_sim,
        'test_intra_similarity': test_intra_sim,
        'inter_set_similarity': inter_sim,
        'inter_similarity_distribution': inter_similarities
    }


def visualize_similarity_analysis(train_df, test_df, similarity_results):
    """
    Create comprehensive visualizations for dataset similarity
    """
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Analysis of Training and Test Set Similarity',
                 fontsize=16, fontweight='bold')

    # 1. Similarity distribution histogram
    axes[0, 0].hist(similarity_results['inter_similarity_distribution'],
                    bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(similarity_results['inter_set_similarity'],
                       color='red', linestyle='--',
                       label=f'Mean similarity: {similarity_results["inter_set_similarity"]:.3f}')
    axes[0, 0].set_xlabel('Tanimoto Similarity Coefficient')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Inter-set Similarity Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 6. Similarity summary
    metrics = ['Train Intra', 'Test Intra', 'Inter-set']
    values = [
        similarity_results['train_intra_similarity'],
        similarity_results['test_intra_similarity'],
        similarity_results['inter_set_similarity']
    ]

    bars = axes[1, 2].bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[1, 2].set_ylabel('Average Similarity')
    axes[1, 2].set_title('Similarity Metrics Summary')
    axes[1, 2].set_ylim(0, 1)

    for bar, value in zip(bars, values):
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dataset_similarity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def advanced_similarity_analysis(train_df, test_df, cat_col='Category'):
    """
    Advanced similarity analysis using t-SNE and clustering
    """
    # 1. Generate fingerprints
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
                 for mol in train_df['mol'] if mol is not None]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
                for mol in test_df['mol'] if mol is not None]

    train_arrays = np.array([np.array(fp) for fp in train_fps])
    test_arrays = np.array([np.array(fp) for fp in test_fps])
    all_arrays = np.vstack([train_arrays, test_arrays])

    # 2. t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(all_arrays)

    # 3. Extract category labels (train + test)
    train_cats = train_df.loc[train_df['mol'].notnull(), cat_col].values
    test_cats = test_df.loc[test_df['mol'].notnull(), cat_col].values
    all_cats = np.hstack([train_cats, test_cats])

    # 4. Plot
    plt.figure(figsize=(14, 6))

    # ---- 4a Color by training/test ----
    plt.subplot(1, 2, 1)
    plt.scatter(tsne_results[:len(train_arrays), 0],
                tsne_results[:len(train_arrays), 1],
                c='blue', alpha=0.6, s=30, label='Training')
    plt.scatter(tsne_results[len(train_arrays):, 0],
                tsne_results[len(train_arrays):, 1],
                c='red', alpha=0.6, s=30, label='Test')
    plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
    plt.title('Chemical Space (train vs test)')
    plt.legend(); plt.grid(True, alpha=0.3)

    # ---- 4b Color by category ----
    plt.subplot(1, 2, 2)
    unique_cats = sorted(np.unique(all_cats))
    palette = sns.color_palette('tab10', n_colors=len(unique_cats))
    color_map = dict(zip(unique_cats, palette))

    for cat in unique_cats:
        mask = all_cats == cat
        plt.scatter(tsne_results[mask, 0],
                    tsne_results[mask, 1],
                    c=[color_map[cat]],
                    label=f'Cat {cat}',
                    alpha=0.7, s=30)
    plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
    plt.title('Chemical Space (by category)')
    plt.legend(title=cat_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('advanced_similarity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# Main execution function
def main(train_file, test_file, smiles_column='SMILES'):
    """
    Main function to execute the complete similarity analysis
    """
    print("Loading and preprocessing data...")
    train_df, test_df = load_and_preprocess_data(train_file, test_file, smiles_column)

    print("Calculating similarity metrics...")
    similarity_results = compute_similarity_metrics(train_df, test_df)

    print("\n=== Similarity Results ===")
    print(f"Training set internal similarity: {similarity_results['train_intra_similarity']:.3f}")
    print(f"Test set internal similarity: {similarity_results['test_intra_similarity']:.3f}")
    print(f"Inter-set similarity: {similarity_results['inter_set_similarity']:.3f}")

    print("Generating visualizations...")
    visualize_similarity_analysis(train_df, test_df, similarity_results)

    print("Performing advanced analysis...")
    advanced_similarity_analysis(train_df, test_df, cat_col='Category')

    return similarity_results, train_df, test_df


if __name__ == "__main__":
    results, train_data, test_data = main('train.csv', 'test.csv', smiles_column='SMILES')
