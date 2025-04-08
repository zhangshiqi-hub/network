import torch
import os
# import datasets
# from datasets import DatasetBuilder, BuilderConfig, GeneratorBasedBuilder
# from datasets.download.download_manager import DownloadManager
# from datasets.info import DatasetInfo
import os.path as osp
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pandas as pd
import configparser
from rdkit import Chem
# from .utils import mol_to_graph_dict
from typing import Literal
# import yaml
# from easydict import EasyDict
import dgl
import torch
from dgl.data import DGLDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

"""
    This module contains utility functions for data processing.
"""

import rdkit
import numpy as np
from typing import Dict, List, Union

ALLOWABLE_FEATURES = {
    # This dictionary contains the allowable features for each node and edge.
    # "possible_atomic_num": list(range(1, 119)) + ["misc"],
    "possible_atomic_num": list(range(1, 119)),
    "possible_chirality": ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
    "possible_degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic": [False, True],
    "possible_is_in_ring": [False, True],
    # "possible_bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
        "possible_bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
        "possible_bond_dirs": ["NONE", "ENDUPRIGHT", "ENDDOWNRIGHT"],
        "possible_bond_stereo": ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"],
        "possible_is_conjugated": [False, True],
}


def safe_index(ls: List, ele: Union[str, int, bool]):
    """Return index of element in the list. If ele is not present, return the last index of the list

    Args:
        ls (List): list of elements
        ele (Union[str, int, bool]): element to find index of
    return:
        index of ele in ls
    """
    try:
        return ls.index(ele)
    except ValueError:
        return len(ls) - 1


def atom_to_feature_vector(atom: rdkit.Chem.rdchem.Atom):
    """Convert RDKit atom to feature vector

    Args:
        atom (rdkit.Chem.rdchem.Atom): RDKit atom

    Returns:
        List: feature vector
    """
    feature_vector = [
        safe_index(ALLOWABLE_FEATURES["possible_atomic_num"], atom.GetAtomicNum()),
        safe_index(ALLOWABLE_FEATURES["possible_chirality"], str(atom.GetChiralTag())),
        safe_index(ALLOWABLE_FEATURES["possible_degree"], atom.GetDegree()),
        safe_index(ALLOWABLE_FEATURES["possible_formal_charge"], atom.GetFormalCharge()),
        safe_index(ALLOWABLE_FEATURES["possible_numH"], atom.GetTotalNumHs()),
        safe_index(ALLOWABLE_FEATURES["possible_number_radical_e"], atom.GetNumRadicalElectrons()),
        safe_index(ALLOWABLE_FEATURES["possible_hybridization"], str(atom.GetHybridization())),
        safe_index(ALLOWABLE_FEATURES["possible_is_aromatic"], atom.GetIsAromatic()),
        safe_index(ALLOWABLE_FEATURES["possible_is_in_ring"], atom.IsInRing()),
    ]
    return feature_vector


def bond_to_feature_vector(bond: rdkit.Chem.rdchem.Bond):
    """Convert RDKit bond to feature vector

    Args:
        bond (rdkit.Chem.rdchem.Bond): RDKit bond

    Returns:
        List: feature vector
    """
    feature_vector = [
        safe_index(ALLOWABLE_FEATURES["possible_bond_type"], str(bond.GetBondType())),
        safe_index(ALLOWABLE_FEATURES["possible_bond_dirs"], str(bond.GetBondDir())),
        safe_index(ALLOWABLE_FEATURES["possible_bond_stereo"], str(bond.GetStereo())),
        safe_index(ALLOWABLE_FEATURES["possible_is_conjugated"], bond.GetIsConjugated()),
    ]
    return feature_vector

# 核心：将分子转化为dict
def mol_to_graph_dict(molecule: rdkit.Chem.rdchem.Mol, properties_dict: Union[Dict[str, float], None] = None):
    """Convert RDKit molecule to a python dictionary containing the graph information (based on the original code of Molecule3D)

    Args:
        molecule (rdkit.Chem.rdchem.Mol): RDKit molecule

    Returns:
        Dict: dictionary containing the graph information
    """
    # smiles = rdkit.Chem.MolToSmiles(molecule)
    try:
        conformer = molecule.GetConformer()  # type: rdkit.Chem.rdchem.Conformer
        # This conformer comes from the sdf file which is the ground_truth conformer added into the sdf file by the author.
        # The way to generate conformers in rdKit is rdkit.Chem.AllChem.EmbedMolecule(mol: rdkit.Chem.rdchem.Mol).
    except ValueError:
        conformer = None

    # Build atom features
    node_attr = [atom_to_feature_vector(atom) for atom in molecule.GetAtoms()]
    # Build atom type
    node_type = [atom_feature[0] for atom_feature in node_attr]
    # Build atom chiral type
    node_chiral_type = [atom_feature[1] for atom_feature in node_attr]

    # Build bond features
    num_bond_features = 3
    if len(molecule.GetBonds()) > 0:
        edges_ls = []
        edges_features_ls = []
        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_ls.append([i, j])
            edges_features_ls.append(edge_feature)
            edges_ls.append([j, i])
            edges_features_ls.append(edge_feature)
        edge_index = np.array(edges_ls, dtype=np.int64).T.tolist()
        edge_attr = np.array(edges_features_ls, dtype=np.int64).tolist()
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64).tolist()
        edge_attr = np.zeros((0, num_bond_features), dtype=np.int64).tolist()
    # Build bond type
    edge_type = [edge_feature[0] for edge_feature in edge_attr]
    # Build bond dire type
    edge_dir_type = [edge_feature[1] for edge_feature in edge_attr]

    # Build graph dictionary
    graph = {
        # "smiles": smiles,
        "node_attr": node_attr,
        "edge_attr": edge_attr,
        "edge_index": edge_index,
        "node_type": node_type,
        "node_chiral_type": node_chiral_type,
        "edge_type": edge_type,
        "edge_dire_type": edge_dir_type,
        "num_nodes": len(node_attr),
        "num_edges": len(edge_attr),
    }
    if conformer is not None:
        graph["conformer"] = conformer.GetPositions().tolist()
    graph.update(properties_dict) if properties_dict is not None else None

    return graph



HAR2EV = 27.211386246  # Hartree to eV
KCALMOL2EV = 0.04336414  # kcal/mol to eV

CONVERSION = {
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
    "mu": 1.0,
    "alpha": 1.0,
    "homo": HAR2EV,
    "lumo": HAR2EV,
    "gap": HAR2EV,
    "r2": 1.0,
    "zpve": HAR2EV,
    "u0": HAR2EV,
    "u298": HAR2EV,
    "h298": HAR2EV,
    "g298": HAR2EV,
    "cv": 1.0,
    "u0_atom": KCALMOL2EV,
    "u298_atom": KCALMOL2EV,
    "h298_atom": KCALMOL2EV,
    "g298_atom": KCALMOL2EV,
}


# _ROOT_URL = "https://huggingface.co/datasets/RichXuOvO/HFQm9/resolve/main"

# _URLS = {
#     "raw_sdf": f"{_ROOT_URL}/gdb9.sdf",
#     "properties_csv": f"{_ROOT_URL}/gdb9.sdf.csv",
#     "train_csv": f"{_ROOT_URL}/train_indices.csv",
#     "valid_csv": f"{_ROOT_URL}/valid_indices.csv",
#     "test_csv": f"{_ROOT_URL}/test_indices.csv",
# }



# 用于将CSv文件里面的数据进行标准化
CONVERSION = {
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
    "mu": 1.0,
    "alpha": 1.0,
    "homo": HAR2EV,
    "lumo": HAR2EV,
    "gap": HAR2EV,
    "r2": 1.0,
    "zpve": HAR2EV,
    "u0": HAR2EV,
    "u298": HAR2EV,
    "h298": HAR2EV,
    "g298": HAR2EV,
    "cv": 1.0,
    "u0_atom": KCALMOL2EV,
    "u298_atom": KCALMOL2EV,
    "h298_atom": KCALMOL2EV,
    "g298_atom": KCALMOL2EV,
}


import torch



def load_QM9(self, dataset_path, mode, standardize= True):
    # '''
    # Args:
    #         standardize (bool, optional): followed https://arxiv.org/abs/2206.11990,
    #         normalized all properties by subtracting the mean and dividing by the Mean Absolute Deviation.
    # '''
    # """
    # Load the QM9
    # """
    properties_csv_path = f"{dataset_path}/gdb9.sdf.csv"
    raw_sdf_path = f"{dataset_path}/gdb9.sdf"

    properties_df = pd.read_csv(properties_csv_path)
    list_of_data=[]
    for column in CONVERSION.keys():  # convert units
        properties_df[column] = properties_df[column] * CONVERSION[column]
    columns = properties_df.drop(columns="mol_id").columns
    if standardize:
        for col in columns:  # standardize properties by subtracting the mean and dividing by the Mean Absolute Deviation
            if col in ["u0_atom", "u298_atom", "h298_atom", "g298_atom"]:
                mean, std = 0, 1
            else:
                mean = properties_df[col].mean()
                std = properties_df[col].std()
            properties_df[col] = (properties_df[col] - mean) / std

    indices_csv_path = f"{dataset_path}/{mode}_indices.csv"
    indices_df = pd.read_csv(indices_csv_path)
    indices = indices_df["index"].values.tolist()
    supplier = Chem.SDMolSupplier(fileName=raw_sdf_path, removeHs=False, sanitize=True)
    for idx in tqdm(indices):
        mol = supplier[idx]
        properties_dict = properties_df.iloc[idx].to_dict()
        if mol is None:
            continue
        graph_dict = mol_to_graph_dict(mol, properties_dict)
        # yield idx, graph_dict
        list_of_data.append((idx, graph_dict))
    return  list_of_data

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels,conformers = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, labels,conformers

class QM9Dataset(DGLDataset):
    def __init__(self,dataset_path,mode, standardize=True,name="QM9", *args, **kwargs):
        """
        初始化数据集，传入分子图数据
        """
        self.dataset_path = dataset_path
        self.mode = mode
        self.standardize =standardize
        super().__init__(name=name,*args, **kwargs)

    def process(self):
        """
        处理并生成DGL图，每个图对应一个分子
        """
        self.data=load_QM9(self,dataset_path=self.dataset_path,mode=self.mode,standardize=self.standardize)# 包含多个分子图的字典
        self.graphs = []  # 存储图对象
        self.properties = []
        self.conformer=[]
        self.mol_id=[]

        for mol_id, mol_data in self.data:

            self.mol_id.append(mol_id)
            # 提取分子图的属性
            node_attr = torch.tensor(mol_data['node_attr'], dtype=torch.float32)
            edge_attr = torch.tensor(mol_data['edge_attr'], dtype=torch.float32)
            edge_index = torch.tensor(mol_data['edge_index'], dtype=torch.long)
            node_type = torch.tensor(mol_data['node_type'], dtype=torch.long)
            node_chiral_type = torch.tensor(mol_data['node_chiral_type'], dtype=torch.long)
            edge_type = torch.tensor(mol_data['edge_type'], dtype=torch.long)
            edge_dire_type = torch.tensor(mol_data['edge_dire_type'], dtype=torch.long)

            # 创建DGL图
            g = dgl.graph((edge_index[0], edge_index[1]))
            g.ndata['feat'] = node_attr  # 节点特征（例如某种原子属性）
            g.ndata['node_type'] = node_type
            g.ndata['node_chiral_type'] = node_chiral_type
            g.edata['edge_attr'] = edge_attr
            g.edata['edge_type'] = edge_type
            g.edata['edge_dire_type'] = edge_dire_type
            # g.ndata['atom_mask'] = torch.ones(node_attr.shape[0], 1, dtype=torch.float32)


            # 其他额外信息
            conformer = torch.tensor(mol_data['conformer'], dtype=torch.float32)
            self.conformer.append(conformer)

            # 存储分子ID作为全局属性

            # A = mol_data["A"]
            # B = mol_data["B"]
            # C = mol_data["C"]
            # mu = mol_data["mu"]
            # alpha = mol_data["alpha"]
            # homo = mol_data['homo']
            # lumo = mol_data['lumo']
            # gap = mol_data['gap']
            # r2=mol_data['r2']
            # zpve=mol_data['zpve']
            # u0=mol_data['u0']
            # u298=mol_data['u298']
            # h298=mol_data['h298']
            # g298=mol_data['g298']
            # h298_atom=mol_data['h298_atom']
            # g298_atom=mol_data['g298_atom']
            # u298_atom=mol_data['u298_atom']
            # cv=mol_data['cv']
            # u0_atom=mol_data['u0_atom']

            _labels = {
                'A': mol_data["A"], 'B': mol_data["B"], 'C': mol_data["C"], 'mu': mol_data["mu"],
                'alpha': mol_data["alpha"],
                'homo': mol_data['homo'], 'lumo': mol_data['lumo'], 'gap': mol_data['gap'], 'r2': mol_data['r2'],
                'zpve': mol_data['zpve'], 'u0': mol_data['u0'], 'u298': mol_data['u298'], 'h298': mol_data['h298'],
                'g298': mol_data['g298'], 'cv': mol_data['cv'], 'u0_atom': mol_data['u0_atom'],
                'u298_atom': mol_data['u298_atom'], 'h298_atom': mol_data['h298_atom'],
                'g298_atom': mol_data['g298_atom']
            }
            # 定义键的顺序
            keys_order = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298',
                          'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']

            # 根据键的顺序创建列表
            labels = torch.tensor([_labels[key] for key in keys_order],dtype=torch.float32)

            # 存储图和相关属性
            self.graphs.append(g)
            self.properties.append(labels)
            # print(mol_id)
            # print(conformer)
            # print(self.graphs)
            # print("done")


    def __getitem__(self, idx):
        """
        获取指定索引的图及其相关属性
        """

        graph = self.graphs[idx]
        label = self.properties[idx]
        conformer = self.conformer[idx]

        return graph, label, conformer


    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.graphs)

    # 假设 `data` 是你提供的分子图数据

if __name__ == "__main__":
    print("load dataset")
    # 创建数据集
    dataset = QM9Dataset(dataset_path="../dataset/QM9", standardize=True, mode="train")
    # 访问数据集中的某个图、标签和分子坐标
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True,
                             collate_fn=collate)

    import networkx as nx
    for batched_graph, labels,conformer in data_loader:
        # batched_graph是一个BatchedDGLGraph对象，包含了当前批次的图
        # labels是当前批次中图的标签
        # 在这里进行训练或评估


        # Since the actual graph is undirected, we convert it for visualization
        # purpose.
        adj_matrix = batched_graph.adjacency_matrix()
        print(adj_matrix)
        print(batched_graph)
        print(len(labels))
        print(len(conformer))




