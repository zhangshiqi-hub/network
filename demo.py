import torch
from torch import Tensor, nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.emb=nn.Embedding(128,128)
        self.emb2=nn.Linear(256,128)
    def _get_atomic_number_message(self, z: Tensor, edge_index: Tensor) -> Tensor:
        Z = self.emb(z)
        Zij = self.emb2(
            Z.index_select(0, edge_index.t().reshape(-1)).view(
            -1, self.hidden_channels * 2
        )
        )[..., None, None]
        return Zij
    def forward(self,z,edge_index):
        Zij = self._get_atomic_number_message(z, edge_index)
        print(z.shape, Zij.shape)

if __name__ == "__main__":
    # z=torch.rand(10,7)# 原子特征(natom,dim)
    # pos=torch.rand(10,3)  # 位置[natom,3]
    # batch=torch.zeros(10,1)
    # adjmatrix = torch.rand(100)  # 用随机数初始化邻接矩阵，仅作示例
    # adjmatrix = (adjmatrix > 0.5).float()  # 将随机数转换为0或1，模拟边的存在
    # model=TensorNet()
    # x, y, z, pos, batch=model(z=z,pos=pos,batch=batch,adjacency=adjmatrix)
    # print(x)
    z=torch.randn(128,1).long()
    edge=torch.randn(2,256)
    net=Net()
    net(z,edge)