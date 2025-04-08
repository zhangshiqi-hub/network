
'''
第二个模型的训练
'''
from datetime import datetime

from Utils.datautils import _compute_loss, ConvertToInput, move_dict_to_device
from Utils.qm9 import *
from Models.GCN import *
from Models.consrtuctformer import *
from Utils.utils import get_edge
from config import Config
import torch

from predict import TensorNet

torch.autograd.set_detect_anomaly(True)

config = Config(1,1,1,nhead=4,dropout=0)
device = torch.device("cuda" if torch.cuda.is_available() else "")
dataset = QM9Dataset(dataset_path="./dataset/QM9", standardize=True, mode="train")
# 访问数据集中的某个图、标签和分子坐标
# model= GraphTransformer(in_feats=9, gcn_out_feats=64, transformer_out_feats=32).to(torch.device("cuda"))
# model=GTMGCForConformerPrediction(config=config).to(device)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True,
                             collate_fn=collate)



model2 = TensorNet(hidden_channels=256, num_layers=4).to(device)
model2.load_state_dict(torch.load(r"D:\Pyprograms\New\checkpoint\2025-03-12_07-16-03\model2_weights.pth"))
# for batched_graph, labels ,conformer in tqdm(data_loader):
#     batched_graph = batched_graph.to(device)
#     # batched_graph.ndata["feat"] = batched_graph.ndata["feat"].to(device)
#
#     # # 确保标签也移动到CUDA设备（如果需要的话）
#     # labels = labels.to(device)
#
#     # 前向传播
#     conformer_hat = model(batched_graph)
#     # print(conformer)
#     conformer=torch.cat(conformer,dim=0)
#     conformer=torch.tensor(conformer).to(device)
#     conformer=conformer.unsqueeze_(0)
#     # conformer_hat=conformer_hat.unsqueeze_(0)
#     conformer_hat=align_conformer_hat_to_conformer(conformer_hat, conformer)
#     loss=_compute_loss(conformer,conformer_hat,torch.tensor(batched_graph.num_nodes()))
#     loss.backward()
#     print(loss.item())
#     # 更新梯度
#     optimizer.step()
#
#     # 清零梯度`
#     optimizer.zero_grad()
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-5)
criterion = nn.MSELoss()



for epoch in range(10):
    for batched_graph, labels, conformer in tqdm(data_loader, desc="Training", leave=True) :        # batched_graph是dgl图 ；label 是分子性质，conformer是分子构象
        optimizer.zero_grad()
        # batched_graph = batched_graph.to(device)
        # conformer = torch.cat(conformer, dim=0)  # n,d
        # conformer = torch.tensor(conformer).to(device)
        # conformer = conformer.unsqueeze_(0) # b,n,d
        batched_graph = batched_graph.to(device)
        conformer = torch.tensor(torch.cat(conformer, dim=0)).unsqueeze_(0).to(device)  # n,d
        # conformer = torch.tensor(conformer).to(device)
        # conformer = conformer.unsqueeze_(0) # b,n,d

        input= ConvertToInput(batched_graph, conformer)
        adj_sparse= get_edge(batched_graph)

        input["adj_sparse"] = adj_sparse
        input["label"]=labels
        input = move_dict_to_device(input, device)
        # 稀疏表示形式

        x,z, pos, batch,out=model2(z=input["node_encodding"],pos=input["conformer"],batch=input["batch"],adjacency=input["adj_sparse"])

        '''
        预测分子坐标
        output = model(input)
        loss=output["loss"]
        loss.backward()
        '''
        #
        # print(conformer)


        loss = criterion(out, torch.stack(input["label"]).to(device))
        loss.backward()
        print(loss.item())





        # # 对齐 conformer_hat 到 conformer
        # conformer_hat = align_conformer_hat_to_conformer(conformer_hat, conformer)

        # 计算损失
        # loss = _compute_loss(conformer, conformer_hat, torch.tensor(batched_graph.num_nodes
        # 打印损失
        # print(loss)

        # 更新梯度
        optimizer.step()



        # 更新进度条，显示实时的 loss 值d
        tqdm.write(f"Loss: {loss.item()}")
    # 保存模型权重
    # 获取当前时间并格式化为文件夹名
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join("checkpoint", current_time)

    # 创建新文件夹
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存模型的权重
    model_save_path = os.path.join(checkpoint_dir, "model2_weights.pth")
    torch.save(model2.state_dict(), model_save_path)

    # 打印保存路径
    print(f"Model saved at: {model_save_path}")