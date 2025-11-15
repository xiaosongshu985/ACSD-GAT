import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem

def calculate_descriptors(input_name):
    # Load data
    data = pd.read_csv(input_name + '.csv', encoding='gbk', header=None)

    # Assume the second column is SMILES, name it 'smiles'
    data.columns = ['ID', 'smiles'] + [f'Col{i}' for i in range(2, data.shape[1])]

    # Generate molecule objects
    data['Mol'] = data['smiles'].apply(Chem.MolFromSmiles)

    # Define calculation functions
    def calculate_morgan(mol, nBits=2048, radius=2):
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            return list(fp)
        return [0] * nBits

    def calculate_rdk(mol, nBits=2048):
        if mol is not None:
            fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=nBits)
            return list(fp)
        return [0] * nBits

    # Calculate RDK and Morgan fingerprints
    data['rdk'] = data['Mol'].apply(calculate_rdk)
    data['morgan'] = data['Mol'].apply(calculate_morgan)

    # Convert to arrays
    descriptors_array1 = np.array(data['rdk'].tolist())
    descriptors_array2 = np.array(data['morgan'].tolist())

    # Calculate MACCS keys
    MACC = np.array([MACCSkeys.GenMACCSKeys(x) if x is not None else [0]*167 for x in data['Mol']])

    # Combine all features
    combined_descriptors = np.hstack((MACC, descriptors_array1, descriptors_array2))

    # Save as CSV
    df = pd.DataFrame(combined_descriptors)
    output_filename = f'fp_spoc__{input_name}.csv'
    df.to_csv(output_filename, index=False)
    print(f"Fingerprint file saved as: {output_filename}")

if __name__ == '__main__':
    input_name = 'all_exp+mtkl-zz'
    calculate_descriptors(input_name)
