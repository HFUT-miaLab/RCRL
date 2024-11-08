import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.GatedAttention import GatedAttention
from models.TransLayer import TransLayer


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim=512, k_dim=512, v_dim=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        # 定义线性投影层，用于将输入变换到多头注意力空间
        self.proj_q = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(in_dim, v_dim * num_heads, bias=False)
        # 定义多头注意力的线性输出层
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, in_dim = x.size()

        q = self.proj_q(x).view(batch_size, seq_len, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k = self.proj_k(x).view(batch_size, seq_len, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v = self.proj_v(x).view(batch_size, seq_len, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        # 计算注意力权重和输出结果
        attn = torch.matmul(q, k) / self.k_dim ** 0.5  # 注意力得分

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)  # 注意力权重参数
        output = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # 输出结果
        # 对多头注意力输出进行线性变换和输出
        output = self.proj_o(output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, in_dim1=512, in_dim2=512, k_dim=512, v_dim=512, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Sequential(nn.Linear(in_dim1, k_dim * num_heads, bias=False))
        self.proj_k2 = nn.Sequential(nn.Linear(in_dim2, k_dim * num_heads, bias=False))
        self.proj_v2 = nn.Sequential(nn.Linear(in_dim2, v_dim * num_heads, bias=False))
        self.proj_o = nn.Sequential(nn.Linear(v_dim * num_heads, in_dim1))

    def forward(self, query, key_value, mask=None):
        batch_size, seq_len1, in_dim1 = query.size()
        seq_len2 = key_value.size(1)
        #[1 N 512]
        q1 = self.proj_q1(query).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(key_value).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(key_value).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output

class Conv_layer(nn.Module):
    def __init__(self, dim=512):
        super(Conv_layer, self).__init__()
        self.proj = nn.Conv1d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv1d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv1d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x):
        x = x.transpose(1,2)
        x = (self.proj(x) + x + self.proj1(x) + self.proj2(x)).transpose(1,2)
        return x


class RCRL(nn.Module):
    def __init__(self, n_classes, max_length = 4, patch_dim=512, text_dim=512, text_tokens_dim=768, ds=512, dt=512, df=512):
        super(RCRL, self).__init__()
        self.n_classes = n_classes
        self.K = 8
        self.max_length = max_length
        self.layer1 = TransLayer(dim=ds)
        self.layer2= TransLayer(dim=ds)
        self.Multihead_attention = MultiHeadAttention()
        self.Cross_attention = CrossAttention()
        self.fc_image = nn.Sequential(nn.Linear(patch_dim, ds), nn.ReLU())
        self.fc_histo_tokens = nn.Sequential(nn.Linear(text_tokens_dim, dt), nn.ReLU())
        self.fc_histo = nn.Sequential(nn.Linear(text_dim, dt), nn.ReLU())
        self.fc_gene = nn.Sequential(nn.Linear(text_dim, dt), nn.ReLU())
        self.classifier = nn.Linear(df, self.n_classes)
        self.fc_fusion = nn.Sequential(nn.Linear(ds+dt, df), nn.ReLU())
        self.norm = nn.LayerNorm(ds)
        self.conv_layer = Conv_layer(dim=ds)
        self.register_buffer('sequence_embedd_list',torch.randn(n_classes, max_length, dt))
        self.register_buffer('fusion_embedd_list', torch.randn(n_classes, max_length, df))
        self.Basemodel_WSI = GatedAttention(n_classes)
        self.Basemodel_Text = GatedAttention(n_classes)
        self.register_buffer('original_histo_embedd_list', torch.randn(n_classes, max_length, dt))
        self.register_buffer('reconstruct_histo_embedd_list', torch.randn(n_classes, max_length, dt))
        # cosine similarity as logits
        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def Contrastive_Learning(self, histo, histo_reconstruct_feat, label):
        if label != self.n_classes - 1:  # 不是最后一个标签,防止越界
            original_histo_contrast_list = torch.cat((self.original_histo_embedd_list[0:label].detach()
                                                      , self.original_histo_embedd_list[label + 1:].detach()))
            reconstruct_histo_contrast_list = torch.cat((self.reconstruct_histo_embedd_list[0:label].detach()
                                                         , self.reconstruct_histo_embedd_list[label + 1:].detach()))
        else:
            original_histo_contrast_list = self.original_histo_embedd_list[0:label].detach()
            reconstruct_histo_contrast_list = self.reconstruct_histo_embedd_list[0:label].detach()


        original_histo_contrast_list = original_histo_contrast_list.view(-1, 512)
        original_histo_contrast_list = torch.cat([original_histo_contrast_list, histo])
        reconstruct_histo_contrast_list = reconstruct_histo_contrast_list.view(-1, 512)
        reconstruct_histo_contrast_list = torch.cat([reconstruct_histo_contrast_list, histo_reconstruct_feat])

        # 归一化
        original_histo_contrast_list = original_histo_contrast_list / original_histo_contrast_list.norm(dim=-1,
                                                                                                        keepdim=True)
        reconstruct_histo_contrast_list = reconstruct_histo_contrast_list / reconstruct_histo_contrast_list.norm(dim=-1,
                                                                                                                 keepdim=True)

        logits_per_original = self.logit_scale1.exp() * original_histo_contrast_list @ reconstruct_histo_contrast_list.t()
        logits_original = logits_per_original[-1]
        original_probs = torch.softmax(logits_original, dim=-1)

        logits_per_reconstruct = logits_per_original.t()
        logits_reconstruct = logits_per_reconstruct[-1]
        reconstruct_probs = torch.softmax(logits_reconstruct, dim=-1)

        return original_probs, reconstruct_probs,  original_histo_contrast_list.size()[0]

    def forward(self, image, histo, histo_tokens, sequence, label):
        # tensor shape:
        # CONCH text_tokens-> [B,127,768], image->[B,N,512], text->[B,512], sequence->[B,512]
        # BERT text_tokens-> [B,512,768],  image->[B,N,384], text->[B,768], sequence->[B,768]

        image = self.fc_image(image)
        histo = self.fc_histo(histo)
        histo_tokens = self.fc_histo_tokens(histo_tokens)
        sequence = self.fc_gene(sequence)

        # WSI Branch
        conv_input = self.layer1(image)
        conv_output = self.conv_layer(conv_input)
        image_output = self.layer2(conv_output)

        # Reconstruct (Text Branch)
        ca_output = self.Cross_attention(self.norm(histo_tokens), self.norm(image))  # out_put [B,N,512]
        histo_reconstruct_feat, _ = self.Basemodel_Text(self.norm(ca_output))

        # Feature fusion
        WSI_feat, A = self.Basemodel_WSI(self.norm(image_output))  # A [n_classes*N]  wsi_FEAT [1,512]
        fusion_feat = torch.cat([WSI_feat, histo_reconstruct_feat], dim=-1)
        fusion_feat = self.fc_fusion(fusion_feat)

        # Report reconstruct align
        original_probs, reconstruct_probs, length_reconstruct = self.Contrastive_Learning(histo, histo_reconstruct_feat, label)

        # update the queue
        self.original_histo_embedd_list[int(label)] = torch.cat([self.original_histo_embedd_list[int(label)][1:], histo], dim=0)
        self.reconstruct_histo_embedd_list[int(label)] = torch.cat([self.reconstruct_histo_embedd_list[int(label)][1:], histo_reconstruct_feat], dim=0)

        # predict
        logits = self.classifier(fusion_feat)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        # molecular semantics align
        fusion_probs, sequence_probs, length_sequence = self.Contrastive_Learning(fusion_feat, sequence, label)

        # update the queue
        self.fusion_embedd_list[int(label)] = torch.cat([self.fusion_embedd_list[int(label)][1:], fusion_feat], dim=0)
        self.sequence_embedd_list[int(label)] = torch.cat([self.sequence_embedd_list[int(label)][1:], sequence], dim=0)


        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'feature': fusion_feat
                          ,'fusion_probs': fusion_probs, 'sequence_probs': sequence_probs, 'length_sequence':length_sequence
                          , 'original_probs': original_probs, 'reconstruct_probs': reconstruct_probs, 'length_reconstruct': length_reconstruct
                        }

        return results_dict
