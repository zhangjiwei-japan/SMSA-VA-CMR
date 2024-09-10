import torch
import torch.nn as nn
from torch.nn import init
from msnet import MSNet
from av_attention import Cross_Attention_Layer,Attention_Layer,Co_Attention_Layer,Multi_Stage_Cross_Attention_Layer
import torch.nn.functional as F
from torch.autograd import Variable

classifier_criterion = nn.CrossEntropyLoss().cuda()

class Classifier(nn.Module):
    def __init__(self,latent_dim=64,out_label=10,kaiming_init=False):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(latent_dim, 64),
            # nn.ReLU(),
            nn.Linear(latent_dim, 32, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(32,out_label, bias=False))
        if kaiming_init:
            self._init_weights_classifier()

    def _init_weights_classifier(self):
        for  m in self._modules:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.classifier(x)
        return x #nn.CrossEntropyLoss()
        # return F.softmax(x, dim=1)
        # return F.log_softmax(x, dim = 1) #F.nll_loss

class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=1024, mid_dim=512, output_dim=128,kaiming_init=False):
        super(ImgNN, self).__init__()
        self.visual_encoder = nn.Sequential(
            nn.Linear(input_dim, mid_dim, bias=False),   # 1024 512
            # nn.LayerNorm(mid_dim),
            # nn.BatchNorm1d(mid_dim, affine=False,track_running_stats=False), #使移动均值和移动方差不起作用
            nn.BatchNorm1d(mid_dim, affine=False), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mid_dim, output_dim, bias=False)# 512 128
        )
        if kaiming_init:
            self._init_weights_img()
    def _init_weights_img(self):
        for  m in self._modules:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.constant_(m.bias, 0.0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    def forward(self, x):
        out = self.visual_encoder(x)

        return out

class AudioNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=128, output_dim=128,kaiming_init=False):
        super(AudioNN, self).__init__()
        self.audio_encoder = nn.Sequential(
            # nn.Linear(input_dim, output_dim),   # 128 128
            # nn.LayerNorm(output_dim),
            # nn.BatchNorm1d(output_dim, affine=False,track_running_stats=False), #使移动均值和移动方差不起作用
            nn.BatchNorm1d(input_dim, affine=False), 
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(input_dim, output_dim, bias=False)    # 128 128
        )
        if kaiming_init:
            self._init_weights_audio()
    def _init_weights_audio(self):
        for  m in self._modules:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.constant_(m.bias, 0.0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    def forward(self, x):
        out = self.audio_encoder(x)
        return out

class cross_modal_net(nn.Module):
    def __init__(self, input_dim=1024, mid_dim=512, out_dim=128, class_num = 10, kaiming_init=True):
        super(cross_modal_net, self).__init__()
        self.msnet_a = MSNet()
        self.msnet_v = MSNet()
        self.kaiming_init = kaiming_init # 模型初始化
        self.co_attention = Co_Attention_Layer(hidden_dim=out_dim)
        self.attention_1 = Attention_Layer(hidden_dim=out_dim)
        self.attention_2 = Attention_Layer(hidden_dim=64)
        self.a_attention = Cross_Attention_Layer(hidden_dim =out_dim)
        self.v_attention = Cross_Attention_Layer(hidden_dim =out_dim)
        self.av_attention = Multi_Stage_Cross_Attention_Layer(hidden_dim=out_dim)
        self.visual_layer = ImgNN(input_dim=input_dim, mid_dim=mid_dim, output_dim=out_dim,kaiming_init=self.kaiming_init)
        self.audio_layer = AudioNN(input_dim=out_dim, output_dim=out_dim,kaiming_init=self.kaiming_init)

        # self.visual_layer_1 = nn.Linear(input_dim, mid_dim, bias=False)
        # self.visual_layer_2 = nn.Linear(mid_dim, out_dim, bias=False)
        self.shared_layer_1 = nn.Linear(out_dim, out_dim, bias=False)
        self.shared_layer_2 = nn.Linear(out_dim*3,out_dim, bias=False)
        self.shared_layer_3 = nn.Linear(out_dim,64, bias=False)
        # self.classifier = nn.Linear(64, class_num, bias=False)
        self.classifier_a = Classifier(latent_dim=64,out_label=class_num,kaiming_init=self.kaiming_init)
        self.classifier_v = Classifier(latent_dim=64,out_label=class_num,kaiming_init=self.kaiming_init)
        # self.classifier.apply(weights_init_classifier)
        self.fig_use_multi_attention = True #True #False #True #False #True 
        self.fig_use_attention_128 = True #False #False #True #True #
        self.fig_use_attention_64 = False #False #不使用最后一个attention model

    def forward(self, visual,audio):
        feature_visual = self.visual_layer(visual) 
        # feature_visual = self.visual_layer_2(feature_visual)
        feature_visual = self.shared_layer_1(feature_visual)
        feature_visual = feature_visual.view(feature_visual.size(0),1,feature_visual.size(1)) # torch.Size([32,1, 128])
        # print(feature_visual.shape)

        feature_audio = self.audio_layer(audio) 
        feature_audio = self.shared_layer_1(feature_audio)
        feature_audio = feature_audio.view(feature_audio.size(0),1,feature_audio.size(1)) # torch.Size([32,1, 128])
        # print(feature_audio.shape)

        audio_out_x1,audio_out_x2,audio_out_x3= self.msnet_a(feature_audio) # torch.Size([32, 128])
        visual_out_x1,visual_out_x2,visual_out_x3= self.msnet_v(feature_visual) # torch.Size([32, 128])
        
        if self.fig_use_multi_attention:
            # print("start")
            out_a_x1,feature_map_a_x1 = self.a_attention(audio_out_x1,visual_out_x1) # torch.Size([32, 128])
            # feature_map_a_x1 = self.shared_layer_2(feature_map_a_x1) # torch.Size([32, 32])
            # a_classifier_x1 = self.classifier(feature_map_a_x1) # torch.Size([32, 10])

            out_v_x1,feature_map_v_x1 = self.v_attention(visual_out_x1,audio_out_x1) 
            # feature_map_v_x1 = self.shared_layer_2(feature_map_v_x1)
            # v_classifier_x1 = self.classifier(feature_map_v_x1)

            out_a_x2,feature_map_a_x2 = self.a_attention(audio_out_x2,visual_out_x2)
            # feature_map_a_x2 = self.shared_layer_2(feature_map_a_x2)
            # a_classifier_x2 = self.classifier(feature_map_a_x2)

            out_v_x2,feature_map_v_x2 = self.v_attention(visual_out_x2,audio_out_x2)
            # feature_map_v_x2 = self.shared_layer_2(feature_map_v_x2)
            # v_classifier_x2 = self.classifier(feature_map_v_x2)

            out_a_x3,feature_map_a_x3 = self.a_attention(audio_out_x3,visual_out_x3)
            # feature_map_a_x3 = self.shared_layer_2(feature_map_a_x3)
            # a_classifier_x3 = self.classifier(feature_map_a_x3)

            out_v_x3,feature_map_v_x3 = self.v_attention(visual_out_x3,audio_out_x3)
            # feature_map_v_x3 = self.shared_layer_2(feature_map_v_x3)
            # v_classifier_x3 = self.classifier(feature_map_v_x3)

            # out_a_x4,feature_map_a_x4 = self.a_attention(audio_out_x4,visual_out_x4)
            # # feature_map_a_x4 = self.shared_layer_2(feature_map_a_x4)
            # # a_classifier_x4 = self.classifier(feature_map_a_x4)

            # out_v_x4,feature_map_v_x4 = self.v_attention(visual_out_x4,audio_out_x4)
            # feature_map_v_x4 = self.shared_layer_2(feature_map_v_x4)
            # v_classifier_x4 = self.classifier(feature_map_v_x4)
        else:
            feature_map_a_x1,feature_map_a_x2,feature_map_a_x3 = audio_out_x1,audio_out_x2,audio_out_x3
            feature_map_v_x1,feature_map_v_x2,feature_map_v_x3 = visual_out_x1,visual_out_x2,visual_out_x3
        # out_feature_a = torch.cat((feature_map_a_x1,feature_map_a_x2,feature_map_a_x3),1)
        # out_feature_v = torch.cat((feature_map_v_x1,feature_map_v_x2,feature_map_v_x3),1)
        out_feature_a_0 = torch.cat((feature_map_a_x1,feature_map_a_x2,feature_map_a_x3),1)
        out_feature_v_0 = torch.cat((feature_map_v_x1,feature_map_v_x2,feature_map_v_x3),1)
        out_feature_a_1 = self.shared_layer_2(out_feature_a_0) # 128
        out_feature_v_1 = self.shared_layer_2(out_feature_v_0) # 128
        
        if self.fig_use_attention_128:
            final_feature_a_0,final_feature_v_0 = self.co_attention(out_feature_a_1,out_feature_v_1)
            final_feature_a_0,final_feature_v_0 = self.av_attention(final_feature_a_0,final_feature_v_0)
            final_feature_a_1 = torch.add(final_feature_a_0,out_feature_a_1)
            final_feature_v_1 = torch.add(final_feature_v_0,out_feature_v_1)
            # shared_attention_1, _ = self.attention_1(torch.add(out_feature_a,out_feature_v))
            # final_feature_a = torch.add(out_feature_a,shared_attention_1)
            # final_feature_v = torch.add(out_feature_v,shared_attention_1)
        else:
            final_feature_a = out_feature_a_1
            final_feature_v = out_feature_v_1
    
        final_feature_a = self.shared_layer_3(final_feature_a_1) # 64
        final_feature_v = self.shared_layer_3(final_feature_v_1)
        if self.fig_use_attention_64:
            # print('use 64-d attention model')
            shared_attention_2, _ = self.attention_2(torch.add(final_feature_a,final_feature_v))
            final_feature_a = torch.add(final_feature_a,shared_attention_2)
            final_feature_v = torch.add(final_feature_v,shared_attention_2) # 32
    
        final_classifier_a = self.classifier_a(final_feature_a)
        final_classifier_v = self.classifier_v(final_feature_v)
   
        # out_classifier_a =torch.cat((a_classifier_x1,a_classifier_x2,a_classifier_x3,a_classifier_x4),1)
        # out_classifier_v =torch.cat((v_classifier_x1,v_classifier_x2,v_classifier_x3,v_classifier_x4),1)
        # return final_feature_v,final_feature_a,torch.log_softmax(final_classifier_v,dim=1),torch.log_softmax(final_classifier_a,dim=1)
        return final_feature_v,final_feature_a,final_classifier_v,final_classifier_a


class cross_modal_base(nn.Module):
    def __init__(self, input_dim= 1024, mid_dim=512, out_dim=128, class_num = 10):
        super(cross_modal_base, self).__init__()
        self.visual_layer_1 = nn.Linear(input_dim, mid_dim, bias=False)
        self.visual_layer_2 = nn.Linear(mid_dim, out_dim, bias=False)
        self.shared_layer = nn.Linear(out_dim, 64, bias=False)
        self.classifier = nn.Linear(64, class_num, bias=False)
    def forward(self, visual,audio):
        feature_visual = self.visual_layer_1(visual) 
        feature_visual = self.visual_layer_2(feature_visual)

        feature_visual = self.shared_layer(feature_visual) 
        final_classifier_v = self.classifier(feature_visual)

        feature_audio = self.shared_layer(audio)
        final_classifier_a = self.classifier(feature_audio)

        return feature_visual,feature_audio,final_classifier_v,final_classifier_a

if __name__ == '__main__':
    x_A = torch.rand(32, 1024) 
    x_B = torch.rand(32, 128) 
    net = cross_modal_net()
    # net = cross_modal_base(input_dim=1024, mid_dim=512, out_dim=128, class_num = 10)
    out_A,out_B,label_A,label_B = net(x_A,x_B)
    print(out_A.shape)
    print(out_B.shape)
    print(label_A.shape)
    print(label_B.shape)









