import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs （分子）
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    
class TripletSimilarity(nn.Module):
    def __init__(self, batch_size, device = t.device("cuda:0" if t.cuda.is_available() else "cpu"), temperature=0.5, margin=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        self.register_buffer("margin", torch.tensor(margin).to(device))
        # self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        self.register_buffer("negatives_mask", (~torch.eye(batch_size, batch_size, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        # print("similarity_matrix:\n", similarity_matrix)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        # print("positives:\n", positives)
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        # print("nominator:\n", nominator)
        new_positives = torch.log(1+ nominator)
        # print("new_positives:\n", new_positives)
        negatives_mask = torch.cat([self.negatives_mask, self.negatives_mask], dim=0)
        negatives_mask = torch.cat([negatives_mask, negatives_mask], dim=1)

        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*batch size, 2*batch size   
        # print("denominator:\n", denominator)
        denominator = torch.sum(denominator, dim=1)
        # print("denominator:\n", denominator)
        new_negatives = torch.log(1+ denominator)
        # print("new_negatives:\n", new_negatives)
        loss_partial = torch.maximum(torch.zeros_like(new_negatives),(new_negatives - new_positives + self.margin))
        # print("loss_partial:\n", loss_partial)
        # loss_partial = -(new_positives-new_negatives+ self.margin)
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    
if __name__ == '__main__':
    loss_func = ContrastiveLoss(batch_size=128)
    loss_func1 = TripletSimilarity(batch_size=128)
    emb_i = torch.rand(128, 64).cuda()
    emb_j = torch.rand(128, 64).cuda()

    loss_contra = loss_func(emb_i, emb_j)
    loss_contra1 = loss_func1(emb_i, emb_j)
    print(loss_contra,loss_contra1)