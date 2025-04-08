from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import math
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import warnings

def make_cdist_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    """Make mask for coordinate pairwise distance from padding mask.

    Args:
        padding_mask (torch.Tensor): Padding mask for the batched input sequences with shape (b, l).

    Returns:
        torch.Tensor: Mask for coordinate pairwise distance with shape (b, l, l).
    """
    padding_mask = padding_mask.unsqueeze(-1)
    mask = padding_mask * padding_mask.transpose(-1, -2)
    return mask


def mask_attention_score(attention_score: torch.Tensor, attention_mask: torch.Tensor, mask_pos_value: float = 0.0):
    """Mask the attention score with the attention_mask || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads

    Args:
        attention_score (torch.Tensor): The attention score with shape (b, h, l, l) or (b, l, l).
        attention_mask (torch.Tensor): The attention mask with shape (b, l) or (b, l, l).
        mask_pos_value (float, optional): The value of the position to be masked. Defaults to 0.0.
    Returns:
        torch.Tensor: The masked attention score with shape (b, h, l, l) or (b, l, l).
    """
    shape = attention_score.shape
    b, h, l, _ = shape if len(shape) == 4 else (shape[0], 1, shape[1], shape[2])
    # (b, l, l) -> (b, 1, l, l) if (b, l, l) else (b, h, l, l)
    attention_score = attention_score.view(b, h, l, l) if len(shape) == 3 else attention_score
    # (b, l) -> (b*h, l)
    attention_mask = attention_mask.repeat_interleave(h, dim=0)  # (b*h, l) or (b*h, l, l)
    attention_mask = attention_mask.view(b, h, -1, l)  # (b, h, 1, l) or (b, h, l, l)
    # attention_mask = (1 - attention_mask) * (-100000.0)
    # attention_score += attention_mask  # (b, h, l, l)
    attention_score = attention_score.masked_fill(attention_mask == mask_pos_value, -10000.0)
    return attention_score.view(shape) if len(shape) == 3 else attention_score
class GaussianSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,
        dtype=torch.float32,
    ):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(
            self.cutoff_lower, self.cutoff_upper, self.num_rbf, dtype=self.dtype
        )
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))
class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances: Tensor) -> Tensor:
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            cutoffs = cutoffs * (distances > self.cutoff_lower)
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            return cutoffs

class ShiftedSoftplus(nn.Module):
    r"""Applies the ShiftedSoftplus function :math:`\text{ShiftedSoftplus}(x) = \frac{1}{\beta} *
    \log(1 + \exp(\beta * x))-\log(2)` element-wise.

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.
    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
class ExpNormalSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,
        dtype=torch.float32,
    ):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(
                -self.cutoff_upper + self.cutoff_lower, dtype=self.dtype
            )
        )
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
            dtype=self.dtype,
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
class Swish(nn.Module):
    """Swish activation function as defined in https://arxiv.org/pdf/1710.05941 :

    .. math::

        \text{Swish}(x) = x \cdot \sigma(\beta x)

    Args:
        beta (float, optional): Scaling factor for Swish activation. Defaults to 1.

    """

    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class OptimizedDistance(torch.nn.Module):
    """Compute the neighbor list for a given cutoff.

    This operation can be placed inside a CUDA graph in some cases.
    In particular, resize_to_fit and check_errors must be False.

    Note that this module returns neighbors such that :math:`r_{ij} \\ge \\text{cutoff_lower}\\quad\\text{and}\\quad r_{ij} < \\text{cutoff_upper}`.

    This function optionally supports periodic boundary conditions with
    arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c` must satisfy
    certain requirements:

    .. code:: python

       a[1] = a[2] = b[2] = 0
       a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff
       a[0] >= 2*b[0]
       a[0] >= 2*c[0]
       b[1] >= 2*c[1]

    These requirements correspond to a particular rotation of the system and
    reduced form of the vectors, as well as the requirement that the cutoff be
    no larger than half the box width.

    Parameters
    ----------
    cutoff_lower : float
        Lower cutoff for the neighbor list.
    cutoff_upper : float
        Upper cutoff for the neighbor list.
    max_num_pairs : int
        Maximum number of pairs to store, if the number of pairs found is less than this, the list is padded with (-1,-1) pairs up to max_num_pairs unless resize_to_fit is True, in which case the list is resized to the actual number of pairs found.
        If the number of pairs found is larger than this, the pairs are randomly sampled. When check_errors is True, an exception is raised in this case.
        If negative, it is interpreted as (minus) the maximum number of neighbors per atom.
    strategy : str
        Strategy to use for computing the neighbor list. Can be one of :code:`["shared", "brute", "cell"]`.

        1. *Shared*: An O(N^2) algorithm that leverages CUDA shared memory, best for large number of particles.
        2. *Brute*: A brute force O(N^2) algorithm, best for small number of particles.
        3. *Cell*:  A cell list algorithm, best for large number of particles, low cutoffs and low batch size.
    box : torch.Tensor, optional
        The vectors defining the periodic box.  This must have shape `(3, 3)` or `(max(batch)+1, 3, 3)` if a ox per sample is desired.
        where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
        If this is omitted, periodic boundary conditions are not applied.
    loop : bool, optional
        Whether to include self-interactions.
        Default: False
    include_transpose : bool, optional
        Whether to include the transpose of the neighbor list.
        Default: True
    resize_to_fit : bool, optional
        Whether to resize the neighbor list to the actual number of pairs found. When False, the list is padded with (-1,-1) pairs up to max_num_pairs
        Default: True
        If this is True the operation is not CUDA graph compatible.
    check_errors : bool, optional
        Whether to check for too many pairs. If this is True the operation is not CUDA graph compatible.
        Default: True
    return_vecs : bool, optional
        Whether to return the distance vectors.
        Default: False
    long_edge_index : bool, optional
        Whether to return edge_index as int64, otherwise int32.
        Default: True
    """

    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_pairs=-32,
        return_vecs=False,
        loop=False,
        strategy="brute",
        include_transpose=True,
        resize_to_fit=True,
        check_errors=True,
        box=None,
        long_edge_index=True,
    ):
        super(OptimizedDistance, self).__init__()
        self.cutoff_upper = cutoff_upper
        self.cutoff_lower = cutoff_lower
        self.max_num_pairs = max_num_pairs
        self.strategy = strategy
        self.box: Optional[Tensor] = box
        self.loop = loop
        self.return_vecs = return_vecs
        self.include_transpose = include_transpose
        self.resize_to_fit = resize_to_fit
        self.use_periodic = True
        if self.box is None:
            self.use_periodic = False
            self.box = torch.empty((0, 0))
            if self.strategy == "cell":
                # Default the box to 3 times the cutoff, really inefficient for the cell list
                lbox = cutoff_upper * 3.0
                self.box = torch.tensor(
                    [[lbox, 0, 0], [0, lbox, 0], [0, 0, lbox]], device="cpu"
                )
        if self.strategy == "cell":
            self.box = self.box.cpu()
        self.check_errors = check_errors
        self.long_edge_index = long_edge_index

    def forward(
        self, pos: Tensor,edge_index,edge_weight,edge_vec, batch: Optional[Tensor] = None, box: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Compute the neighbor list for a given cutoff.

        Parameters
        ----------
        pos : torch.Tensor
            A tensor with shape (N, 3) representing the positions.
        batch : torch.Tensor, optional
            A tensor with shape (N,). Defaults to None.
        box : torch.Tensor, optional
            The vectors defining the periodic box.  This must have shape `(3, 3)` or `(max(batch)+1, 3, 3)`,
        Returns
        -------
        edge_index : torch.Tensor
            List of neighbors for each atom in the batch.
            Shape is (2, num_found_pairs) or (2, max_num_pairs).
        edge_weight : torch.Tensor
            List of distances for each atom in the batch.
            Shape is (num_found_pairs,) or (max_num_pairs,).
        edge_vec : torch.Tensor, optional
            List of distance vectors for each atom in the batch.
            Shape is (num_found_pairs, 3) or (max_num_pairs, 3).

        Notes
        -----
        If `resize_to_fit` is True, the tensors will be trimmed to the actual number of pairs found.
        Otherwise, the tensors will have size `max_num_pairs`, with neighbor pairs (-1, -1) at the end.
        """
        use_periodic = self.use_periodic
        if not use_periodic:
            use_periodic = box is not None
        box = self.box if box is None else box
        assert box is not None, "Box must be provided"
        box = box.to(pos.dtype)
        max_pairs: int = self.max_num_pairs
        if self.max_num_pairs < 0:
            max_pairs = -self.max_num_pairs * pos.shape[0]
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        # 这里的实现有问题，这个函数需要重新实现
        # edge_index, edge_vec, edge_weight, num_pairs = get_neighbor_pairs(
        #     pos=pos,
        #     batch=batch,
        #     max_num_pairs=int(max_pairs),
        #     cutoff=self.cutoff_upper,
        #
        # )
        #直接用传递的邻接矩阵解决

        if self.check_errors:
            assert (
                num_pairs[0] <= max_pairs
            ), f"Found num_pairs({num_pairs[0]}) > max_num_pairs({max_pairs})"

        # Remove (-1,-1)  pairs
        if self.resize_to_fit:
            mask = edge_index[0] != -1
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_vec = edge_vec[mask, :]
        if self.long_edge_index:
            edge_index = edge_index.to(torch.long)
        if self.return_vecs:
            return edge_index, edge_weight, edge_vec
        else:
            return edge_index, edge_weight, None

rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}
act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "swish": Swish,
    "mish": nn.Mish,
}
# def get_neighbor_pairs_kernel(
#     strategy: str,
#     positions: Tensor,
#     batch: Tensor,
#     box_vectors: Tensor,
#     use_periodic: bool,
#     cutoff_lower: float,
#     cutoff_upper: float,
#     max_num_pairs: int,
#     loop: bool,
#     include_transpose: bool,
# ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
#     """Computes the neighbor pairs for a given set of atomic positions.
#     The list is generated as a list of pairs (i,j) without any enforced ordering.
#     The list is padded with -1 to the maximum number of pairs.
#
#     Parameters
#     ----------
#     strategy : str
#         Strategy to use for computing the neighbor list. Can be one of :code:`["shared", "brute", "cell"]`.
#     positions : Tensor
#         A tensor with shape (N, 3) representing the atomic positions.
#     batch : Tensor
#         A tensor with shape (N,). Specifies the batch for each atom.
#     box_vectors : Tensor
#         The vectors defining the periodic box with shape `(3, 3)` or `(max(batch)+1, 3, 3)` if a different box is used for each sample.
#     use_periodic : bool
#         Whether to apply periodic boundary conditions.
#     cutoff_lower : float
#         Lower cutoff for the neighbor list.
#     cutoff_upper : float
#         Upper cutoff for the neighbor list.
#     max_num_pairs : int
#         Maximum number of pairs to store.
#     loop : bool
#         Whether to include self-interactions.
#     include_transpose : bool
#         Whether to include the transpose of the neighbor list (pair i,j and pair j,i).
#
#     Returns
#     -------
#     neighbors : Tensor
#         List of neighbors for each atom. Shape (2, max_num_pairs).
#     distances : Tensor
#         List of distances for each atom. Shape (max_num_pairs,).
#     distance_vecs : Tensor
#         List of distance vectors for each atom. Shape (max_num_pairs, 3).
#     num_pairs : Tensor
#         The number of pairs found.
#     """
#     return torch.ops.torchmdnet_extensions.get_neighbor_pairs(
#         strategy,
#         positions,
#         batch,
#         box_vectors,
#         use_periodic,
#         cutoff_lower,
#         cutoff_upper,
#         max_num_pairs,
#         loop,
#         include_transpose,
#     )


def get_edge(batched_graph):
    src, dst = batched_graph.edges()

    # 将边的起点和终点转换为张量 [2, edge]
    edge_tensor = torch.stack([src, dst])

    return (edge_tensor)


def compute_distance(pos_i, pos_j, box_vectors, use_periodic):
    # 计算两个原子之间的距离，考虑周期性边界条件
    if use_periodic:
        # 应用最小图像约定
        displacement = pos_i - pos_j
        displacement -= torch.round(displacement / box_vectors) * box_vectors
        distance = torch.norm(displacement, dim=-1)
    else:
        distance = torch.norm(pos_i - pos_j, dim=-1)
    return distance


def get_neighbor_pairs(pos, batch, cutoff, max_num_pairs, use_periodic=False, box_vectors=None):
    natoms = pos.size(0)
    device = pos.device

    # 初始化输出张量
    edge_index = torch.full((2, max_num_pairs), -1, dtype=torch.long, device=device)
    edge_weight = torch.full((max_num_pairs,), -1.0, dtype=torch.float, device=device)
    edge_vec = torch.full((max_num_pairs, 3), -1.0, dtype=torch.float, device=device)

    num_pairs = 0

    # 计算所有原子对之间的距离
    for i in range(natoms):
        for j in range(natoms):
            if i != j and batch[i] == batch[j]:  # 同一个分子内的原子对
                distance = compute_distance(pos[i], pos[j], box_vectors, use_periodic)
                if distance <= cutoff:
                    if num_pairs < max_num_pairs:
                        edge_index[0, num_pairs] = i
                        edge_index[1, num_pairs] = j
                        edge_weight[num_pairs] = distance
                        edge_vec[num_pairs] = pos[i] - pos[j]
                        num_pairs += 1

    # 如果找到的邻居对少于最大邻居数，则截断张量
    if num_pairs < max_num_pairs:
        edge_index = edge_index[:, :num_pairs]
        edge_weight = edge_weight[:num_pairs]
        edge_vec = edge_vec[:num_pairs]

    return edge_index, edge_vec, edge_weight, num_pairs


# 示例用法
natoms = 10  # 假设有10个原子
pos = torch.rand((natoms, 3))  # 随机生成原子位置
batch = torch.randint(0, 2, (natoms, 1))  # 假设有两个分子
cutoff = 1.0  # 截止距离
max_num_pairs = 20  # 最大邻居数

# 调用函数
edge_index, edge_vec, edge_weight, num_pairs = get_neighbor_pairs(pos, batch, cutoff, max_num_pairs)
