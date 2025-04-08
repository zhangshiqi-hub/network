import numpy as np
import torch
from scipy.constants import physical_constants
from torch.utils.data import Dataset
import torch.nn.functional as F
import dgl

from Utils.random_walk import random_walk_coding
from Utils.utils import make_cdist_mask

hartree2eV = physical_constants['hartree-electron volt relationship'][0]
DTYPE = np.float32
DTYPE_INT = np.int32


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q

class QM9Dataset(Dataset):
    """QM9 dataset."""
    num_bonds = 4
    atom_feature_size = 6
    input_keys = ['mol_id', 'num_atoms', 'num_bonds', 'x', 'one_hot',
                  'atomic_numbers', 'edge']
    unit_conversion = {'mu': 1.0,
                       'alpha': 1.0,
                       'homo': hartree2eV,
                       'lumo': hartree2eV,
                       'gap': hartree2eV,
                       'r2': 1.0,
                       'zpve': hartree2eV,
                       'u0': hartree2eV,
                       'u298': hartree2eV,
                       'h298': hartree2eV,
                       'g298': hartree2eV,
                       'cv': 1.0}

    def __init__(self, file_address: str, task: str, mode: str = 'train',
                 transform=None, fully_connected: bool = False):
        """Create a dataset object

        Args:
            file_address: path to data
            task: target task ["homo", ...]
            mode: [train/val/test] mode
            transform: data augmentation functions
            fully_connected: return a fully connected graph
        """
        self.file_address = file_address
        self.task = task
        self.mode = mode
        self.transform = transform
        self.fully_connected = fully_connected

        # Encode and extra bond type for fully connected graphs
        self.num_bonds += fully_connected

        self.load_data()
        self.len = len(self.targets)
        print(f"Loaded {mode}-set, task: {task}, source: {self.file_address}, length: {len(self)}")

    def __len__(self):
        return self.len

    def load_data(self):
        # Load dict and select train/valid/test split
        data = torch.load(self.file_address)
        data = data[self.mode]

        # Filter out the inputs
        self.inputs = {key: data[key] for key in self.input_keys}

        # Filter out the targets and population stats
        self.targets = data[self.task]

        # TODO: use the training stats unlike the other papers
        self.mean = np.mean(self.targets)
        self.std = np.std(self.targets)

    def get_target(self, idx, normalize=True):
        target = self.targets[idx]
        if normalize:
            target = (target - self.mean) / self.std
        return target

    def norm2units(self, x, denormalize=True, center=True):
        # Convert from normalized to QM9 representation
        if denormalize:
            x = x * self.std
            # Add the mean: not necessary for error computations
            if not center:
                x += self.mean
        x = self.unit_conversion[self.task] * x
        return x

    def to_one_hot(self, data, num_classes):
        one_hot = np.zeros(list(data.shape) + [num_classes])
        one_hot[np.arange(len(data)), data] = 1
        return one_hot

    def _get_adjacency(self, n_atoms):
        # Adjust adjacency structure
        seq = np.arange(n_atoms)
        src = seq[:, None] * np.ones((1, n_atoms), dtype=np.int32)
        dst = src.T
        ## Remove diagonals and reshape
        src[seq, seq] = -1
        dst[seq, seq] = -1
        src, dst = src.reshape(-1), dst.reshape(-1)
        src, dst = src[src > -1], dst[dst > -1]

        return src, dst

    def get(self, key, idx):
        return self.inputs[key][idx]

    def connect_fully(self, edges, num_atoms):
        """Convert to a fully connected graph"""
        # Initialize all edges: no self-edges
        adjacency = {}
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    adjacency[(i, j)] = self.num_bonds - 1

        # Add bonded edges
        for idx in range(edges.shape[0]):
            adjacency[(edges[idx, 0], edges[idx, 1])] = edges[idx, 2]
            adjacency[(edges[idx, 1], edges[idx, 0])] = edges[idx, 2]

        # Convert to numpy arrays
        src = []
        dst = []
        w = []
        for edge, weight in adjacency.items():
            src.append(edge[0])
            dst.append(edge[1])
            w.append(weight)

        return np.array(src), np.array(dst), np.array(w)

    def connect_partially(self, edge):
        src = np.concatenate([edge[:, 0], edge[:, 1]])
        dst = np.concatenate([edge[:, 1], edge[:, 0]])
        w = np.concatenate([edge[:, 2], edge[:, 2]])
        return src, dst, w

    def __getitem__(self, idx):
        # Load node features
        num_atoms = self.get('num_atoms', idx)
        x = self.get('x', idx)[:num_atoms].astype(DTYPE)
        one_hot = self.get('one_hot', idx)[:num_atoms].astype(DTYPE)
        atomic_numbers = self.get('atomic_numbers', idx)[:num_atoms].astype(DTYPE)

        # Load edge features
        num_bonds = self.get('num_bonds', idx)
        edge = self.get('edge', idx)[:num_bonds]
        edge = np.asarray(edge, dtype=DTYPE_INT)

        # Load target
        y = self.get_target(idx, normalize=True).astype(DTYPE)
        y = np.array([y])

        # Augmentation on the coordinates
        if self.transform:
            x = self.transform(x).astype(DTYPE)

        # Create nodes
        if self.fully_connected:
            src, dst, w = self.connect_fully(edge, num_atoms)
        else:
            src, dst, w = self.connect_partially(edge)
        w = self.to_one_hot(w, self.num_bonds).astype(DTYPE)

        # Create graph
        G = dgl.DGLGraph((src, dst))

        # Add node features to graph
        G.ndata['x'] = torch.tensor(x)  # [num_atoms,3]
        G.ndata['f'] = torch.tensor(np.concatenate([one_hot, atomic_numbers], -1)[..., None])  # [num_atoms,6,1]

        # Add edge features to graph
        G.edata['d'] = torch.tensor(x[dst] - x[src])  # [num_atoms,3]
        G.edata['w'] = torch.tensor(w)  # [num_atoms,4]

        return G, y

def align_conformer_hat_to_conformer(conformer_hat, conformer):
    """Align conformer_hat to conformer using Kabsch algorithm.

    Args:
        - conformer_hat (torch.Tensor): The conformer to be aligned, with shape (b, l, 3).
        - conformer (torch.Tensor): The reference conformer, with shape (b, l, 3).

    Returns:
        - conformer_hat_aligned (torch.Tensor): The aligned conformer, with shape (b, l, 3).
    """

    if torch.isnan(conformer_hat).any() or torch.isinf(conformer_hat).any():
        print("NaN or Inf detected in conformer_hat")
    if torch.isnan(conformer).any() or torch.isinf(conformer).any():
        print("NaN or Inf detected in conformer")
    # compute the mean of conformer_hat and conformer, and then center them

    epsilon = 1e-8  # 添加一个小的正则化项，避免零向量
    p_mean = conformer_hat.mean(dim=1, keepdim=True)+epsilon
    q_mean = conformer.mean(dim=1, keepdim=True)+epsilon

    p_centered = conformer_hat - p_mean
    q_centered = conformer - q_mean
    # compute the rotation matrix using Kabsch algorithm
    H = torch.matmul(q_centered.transpose(1, 2), p_centered).float()  # shape: (b, 3, 3)
    U, S, V = torch.svd(H)  # shape: (b, 3, 3)
    R = torch.matmul(V, U.transpose(1, 2))  # shape: (b, 3, 3)
    # rotate p_centered using R
    p_rotated = torch.matmul(R, p_centered.transpose(1, 2))  # shape: (b, 3, l)
    p_rotated = p_rotated.transpose(1, 2)  # shape: (b, l, 3)
    # align p_rotated to the spatial position of q
    conformer_hat_aligned = p_rotated + q_mean  # shape: (b, l, 3)
    return conformer_hat_aligned


def _compute_loss(conformer: torch.Tensor, conformer_hat: torch.Tensor) -> dict:
    """
    计算给定 conformer 和 conformer_hat 之间的各种损失指标，并根据原子数归一化。

    参数:
    - conformer (torch.Tensor): 形状为 (b, n, 3)，表示真实的构象。
    - conformer_hat (torch.Tensor): 形状为 (b, n, 3)，表示预测的构象。

    返回:
    - dict: 包含总损失和各类指标（如 cdist_mae, cdist_mse, rmsd 等）的字典。
    """
    # 确保输入形状相同
    assert conformer.shape == conformer_hat.shape, "conformer and conformer_hat must have the same shape"

    # 根据第二个维度计算有效原子数 nnodes
    nnodes = conformer.shape[1]  # n，即每个分子的原子数

    # 计算 MAE (Mean Absolute Error)
    cdist_mae = torch.mean(torch.abs(conformer - conformer_hat), dim=[1, 2])  # 按分子计算绝对误差
    # cdist_mae = torch.mean(cdist_mae) / nnodes  # 归一化到原子数

    # 计算 MSE (Mean Squared Error)
    cdist_mse = torch.mean((conformer - conformer_hat) ** 2, dim=[1, 2])  # 按分子计算平方误差
    # cdist_mse = torch.mean(cdist_mse) / nnodes  # 归一化到原子数

    # 计算 RMSD (Root Mean Square Deviation)
    rmsd = torch.sqrt(torch.mean((conformer - conformer_hat) ** 2, dim=[1, 2]))  # 每个分子的 RMSD
    # rmsd = torch.mean(rmsd) / nnodes  # 归一化到原子数

    # 总损失为加权和（可根据需要调整权重）
    loss = cdist_mse

    # 返回字典
    return {
        "loss": loss,
        "cdist_mae": cdist_mae.detach(),
        "cdist_mse": cdist_mse.detach(),
        "coord_rmsd": rmsd.detach(),
        "conformer": conformer.detach(),
        "conformer_hat": conformer_hat.detach(),
    }

def _compute_cdist_mae(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, nnodes: torch.Tensor) -> torch.Tensor:
    """Compute mean absolute error of conformer and conformer_hat.

    Args:
        - masked_cdist (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer.
        - masked_cdist_hat (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer_hat.
        - cdist_mask (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the mask of the pairwise distance matrix.

    Returns:
        torch.Tensor: The mean absolute error of conformer and conformer_hat.
    """
    mae = F.l1_loss(masked_cdist, masked_cdist_hat, reduction="sum") / nnodes  # exclude padding atoms
    return mae

def compute_distance_residual_bias(cdist: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    # 距离矩阵归一化
    D_max, _ = torch.max(cdist.view(cdist.shape[0], -1), dim=-1, keepdim=True)  # 每个样本的最大值
    D_normalized = cdist / (D_max.view(-1, 1, 1) + epsilon)  # 避免除以零

    # 确保非零距离，避免 log(0) 计算报错
    D_normalized = D_normalized + epsilon

    # 计算 \frac{1}{\log D}
    D_result = 1.0 / torch.log(D_normalized)

    # 将对角线置为 0
    D_result.diagonal(dim1=-2, dim2=-1)[:] = 0
    return D_result

def new_compute_distance_residual(cdist: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    # 步骤 1: 归一化距离矩阵
    D_max, _ = torch.max(cdist.view(cdist.shape[0], -1), dim=-1, keepdim=True)  # 每个样本的最大值
    D_normalized = cdist / (D_max.view(-1, 1, 1) + epsilon)  # 避免除以零

    # 步骤 2: 确保非零距离，避免 log(0) 计算报错
    D_normalized = D_normalized + epsilon

    # 步骤 3: 计算 1 / |log(D_normalized)|
    D_log_abs = torch.abs(torch.log(D_normalized))  # 计算 |log(D_normalized)|
    D_result = 1.0 / D_log_abs  # 取倒数

    # 将对角线置为 0
    D_result.diagonal(dim1=-2, dim2=-1)[:] = 0

    return D_result



def ConvertToInput(graph,conformer):
    node_encodding=graph.ndata['feat'].unsqueeze(0)
    adj=graph.adjacency_matrix().to_dense().unsqueeze(0)
    input={"node_encodding":node_encodding,
           "adjacency":adj,
           "conformer":conformer}
    node_mask = torch.ones((1, node_encodding.shape[1], 1), dtype=torch.int64)
    input["node_mask"] = node_mask
    input["position_encodding"] = random_walk_coding(graph)
    # 获取batch
    batch_sizes = graph.batch_num_nodes()

    # 生成节点所属图的索引
    # 例如，第一个图有3个节点，第二个图有4个节点，创建两个张量 [0, 0, 0] 和 [1, 1, 1, 1]
    batch_tensor = []
    current_idx = 0
    for i, size in enumerate(batch_sizes):
        batch_tensor.append(torch.full((size,), i))  # 每个图的节点都属于该图
        current_idx += size

    # 合并所有张量
    batch_tensor = torch.cat(batch_tensor).view(-1, 1)
    input["batch"] =batch_tensor
    return input


def move_dict_to_device(input_dict, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
