import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Attention_Layer(nn.Module):
    #用来实现mask-attention layer
    def __init__(self, hidden_dim):
        super(Attention_Layer,self).__init__()
        
        self.hidden_dim = hidden_dim

        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
                 
    def forward(self, inputs):
        
        #计算生成QKV矩阵
        Q = self.Q_linear(inputs) 
        # K = self.K_linear(inputs).permute(0, 2, 1)#先进行一次转置
        K = torch.transpose(self.K_linear(inputs),1,0)#先进行一次转置
        V = self.V_linear(inputs) 
        #下面开始计算啦
        alpha = torch.matmul(Q, K)
        #下面开始softmax
        alpha = F.softmax(alpha, dim = 1)
        #print('\nalpha is :', alpha)
        out = torch.matmul(alpha, V)
        feature_map = torch.add(out,inputs)
        # return feature_map
        return out,feature_map

class Co_Attention_Layer(nn.Module):
    #用来实现mask-attention layer
    def __init__(self, hidden_dim):
        super(Co_Attention_Layer,self).__init__()
        
        self.hidden_dim = hidden_dim

        self.Q_v_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.Q_a_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.W_ac_linear = nn.Linear(256, hidden_dim, bias = False)
        self.W_vc_linear = nn.Linear(256, hidden_dim, bias = False)
        self.K_c_linear = nn.Linear(256, hidden_dim, bias = False)
        self.V_c_linear = nn.Linear(256, 256, bias = False)
        self.t_v = nn.Parameter(torch.ones([])) 
        self.t_a = nn.Parameter(torch.ones([])) 

                 
    def forward(self, audio,img ):
        common = torch.cat((audio,img),dim=1)
        #计算生成QKV矩阵
        Q_a = self.Q_a_linear(audio) 
        Q_v = self.Q_a_linear(img) 
        # K = self.K_linear(inputs).permute(0, 2, 1)#先进行一次转置
        K_c = torch.transpose(self.K_c_linear(common),1,0)#先进行一次转置
        V_c = self.V_c_linear(common) 
        #下面开始计算啦
        alpha_ac = torch.matmul(Q_a, K_c)
        alpha_vc = torch.matmul(Q_v, K_c)
        #下面开始softmax
        alpha_ac = F.softmax(alpha_ac, dim = 1)
        alpha_vc = F.softmax(alpha_vc, dim = 1)
        #print('\nalpha is :', alpha)
        out_ac = torch.matmul(alpha_ac, V_c)
        out_vc = torch.matmul(alpha_vc, V_c)
        feature_map_audio = torch.add(self.t_a* self.W_ac_linear(out_ac),audio)
        feature_map_img = torch.add(self.t_v* self.W_vc_linear(out_vc),img)
        # return feature_map
        return feature_map_audio,feature_map_img

class Multi_Stage_Cross_Attention_Layer(nn.Module):
    #用来实现mask-attention layer
    def __init__(self, hidden_dim):
        super(Multi_Stage_Cross_Attention_Layer,self).__init__()
        
        self.hidden_dim = hidden_dim
        self.B_a = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.A_a = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.B_v = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.A_v = nn.Linear(hidden_dim, hidden_dim, bias = False)
                 
    def forward(self, inputs_a,inputs_v):
        
        #计算生成QKV矩阵
        a_a = self.A_a(inputs_a)
        b_a = self.B_a(inputs_a) 
        a_v = torch.transpose(self.A_v(inputs_v),1,0)#先进行一次转置
        b_v = self.B_v(inputs_v)
        c_c_av = torch.matmul(b_a,a_v)

        #下面开始计算啦
        alpha_av = F.softmax(c_c_av, dim = 1)
        att_avv = torch.matmul(alpha_av, b_v)
        att_vaa = torch.matmul(torch.transpose(alpha_av,1,0), a_a)
        feature_map_a = torch.add(att_avv,inputs_a)
        feature_map_v = torch.add(att_vaa,inputs_v)
        return feature_map_a,feature_map_v
      
class Cross_Attention_Layer(nn.Module):
    #用来实现mask-attention layer
    def __init__(self, hidden_dim):
        super(Cross_Attention_Layer,self).__init__()
        
        self.hidden_dim = hidden_dim
        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.M_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
                 
    def forward(self, inputs_a,inputs_v):
        
        #计算生成QKV矩阵
        Q_a = self.Q_linear(inputs_a) 
        K_a = torch.transpose(self.K_linear(inputs_a),1,0)#先进行一次转置
        V_a = self.V_linear(inputs_a) 

        Feature_av = torch.add(inputs_a,inputs_v)
        M_av = torch.transpose(self.M_linear(Feature_av),1,0)#先进行一次转置
        
        alpha_aa = torch.matmul(Q_a, K_a)
        alpha_av = torch.matmul(Q_a, M_av)
        alpha_fusion = torch.add(alpha_aa,torch.transpose(alpha_av,1,0))

        #下面开始计算啦
        alpha_fusion = F.softmax(alpha_fusion, dim = 1)
        out_attention = torch.matmul(alpha_fusion, V_a)
        feature_map = torch.add(out_attention,inputs_a)
        return out_attention,feature_map

if __name__ == '__main__':
    x_A = torch.rand(32, 128) 
    x_B = torch.rand(32, 128) 
    a_attention_net = Cross_Attention_Layer(hidden_dim =128)
    v_attention_net = Cross_Attention_Layer(hidden_dim =128)
    a_v_attention_net = Multi_Stage_Cross_Attention_Layer(hidden_dim =128)
    out_a,feature_map_a = a_attention_net(x_A,x_B)
    out_v,feature_map_v = v_attention_net(x_B,x_A)
    out_a_,out_v_ = a_v_attention_net(x_A,x_B)
    print('\nout_a is :', x_A,feature_map_a,out_a.shape)
    print('\nout_v is :', x_B,feature_map_v,out_v.shape)
    print(out_a_.shape,out_v_.shape)

