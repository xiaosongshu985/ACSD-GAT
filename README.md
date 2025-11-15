## About The Project

Atropisomers play a central role in asymmetric synthesis, drug discovery, and functional material design. However, rational design of atropisomers has long been challenged by the difficulty of predicting their configurational stability, which is governed by the rotational energy barrier (ΔG‡).

To address this issue, we introduce ACSD-GAT, a deep learning framework specifically designed for accurate prediction of atropisomeric rotational barriers. Our approach is built upon two key components:

1.A newly curated benchmark dataset consisting of 1,015 experimentally measured rotational barriers, the largest and most comprehensive dataset available to date.

2.A physics-informed Axial Chirality Structure Descriptor (ACSD) that explicitly quantifies both static and dynamic steric repulsion throughout the rotational process.

By integrating ACSD with a Graph Attention Network (GAT) architecture, ACSD-GAT achieves excellent predictive performance, reaching R² = 0.91 and RMSE = 2.02 kcal/mol on the test set.
The robustness and real-world applicability of the model are further demonstrated through validation on pharmaceuticals, molecular switches, and newly synthesized atropisomers.

## Installation

We recommend using Conda to manage the virtual environment and dependencies.

```bash
# Clone this repository
git clone https://github.com/xiaosongshu985/ACSD-GAT.git
cd script

# Create and activate conda environment
conda create --name ACSDGAT python=3.12 -y  
conda activate ACSDGAT

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

For a quick demonstration of ACSD-GAT capabilities:


# 1. Generate ACSD descriptors 
```sh
    cd ACSD-GAT/script
    jupyter lab 
    python ACSD-GAT/script/fingerprint_descriptors.py
```

You need to know in advance the sequence numbers of the atoms on both sides of the rotation axis as well as the SMILES code of the molecule.
If you want more accurate overlap volume data, you can increase num_samples and num_iterations, but this will also raise the computational cost.

# 2. Train a model 
```sh
python ACSD-GAT/script/ACSD-GAT.py 
```

# 3. Study based on clustering of molecular frameworks
```sh
python ACSD-GAT/script/cluster_molecules.py  #Divided into 20 categories, select two of them as the test set.
python ACSD-GAT/script/chemcompare/chemcompare.py  #Need train.csv and test.csv
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{ACSD-GAT2025,
  title={Physicochemically Informed Axial Chirality Descriptors Enable Accurate Prediction of Atropisomeric Stability}
}
```






