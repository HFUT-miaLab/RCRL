import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class GatedAttention(nn.Module):
    def __init__(self,n_classes):
        super(GatedAttention, self).__init__()
        # 全连接层的隐含单元
        self.L = 512 #512
        self.D = 128
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),  #（ in_dimension, out，dimension）
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, x)  # KxL

        return M , A # Y_prob Y_hat A

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


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
    def __init__(self, n_classes, max_length = 4, patch_dim=512, text_dim=512, text_tokens_dim=768):
        super(RCRL, self).__init__()
        self.n_classes = n_classes
        self.K = 8
        self.max_length = max_length
        self.layer1 = TransLayer(dim=512)
        self.layer2= TransLayer(dim=512)
        self.Multihead_attention = MultiHeadAttention()
        self.Cross_attention = CrossAttention()
        self.fc_image = nn.Sequential(nn.Linear(patch_dim, 512), nn.ReLU())
        self.fc_histo_tokens = nn.Sequential(nn.Linear(text_tokens_dim, 512), nn.ReLU())
        self.fc_histo = nn.Sequential(nn.Linear(text_dim, 512), nn.ReLU())
        self.fc_gene = nn.Sequential(nn.Linear(text_dim, 512), nn.ReLU())
        self.classifier = nn.Linear(512, self.n_classes)
        self.fc_fusion = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.fc7 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.norm = nn.LayerNorm(512)
        self.conv_layer = Conv_layer(dim=512)
        self.register_buffer('sequence_embedd_list',torch.randn(n_classes, max_length, 512))
        self.register_buffer('fusion_embedd_list', torch.randn(n_classes, max_length, 512))
        self.Basemodel_WSI = GatedAttention(n_classes)
        self.Basemodel_Text = GatedAttention(n_classes)
        self.register_buffer('original_histo_embedd_list', torch.randn(n_classes, max_length, 512))
        self.register_buffer('reconstruct_histo_embedd_list', torch.randn(n_classes, max_length, 512))
        # cosine similarity as logits
        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, histo, histo_tokens, sequence, label):
        # tensor shape:
        # CONCH text_tokens-> [B,127,768], image->[B,N,512], text->[B,512], sequence->[B,512]
        # BERT text_tokens-> [B,512,768],  image->[B,N,384], text->[B,768], sequence->[B,768]

        image = self.fc_image(image)
        histo = self.fc_histo(histo)
        histo_tokens = self.fc_histo_tokens(histo_tokens)
        sequence = self.fc_gene(sequence)

        # feature fusion
        conv_input = self.layer1(image)
        conv_output = self.conv_layer(conv_input)
        image_output = self.layer2(conv_output)

        # Reconstruct
        ca_output = self.Cross_attention(self.norm(histo_tokens), self.norm(image))  # out_put [1,N,512]
        histo_reconstruct_feat, _ = self.Basemodel_Text(self.norm(ca_output))

        # prompt-guided aggregation
        WSI_feat, A = self.Basemodel_WSI(self.norm(image_output))  # A [n_classes*N]  wsi_FEAT [1,512]
        fusion_feat = torch.cat([WSI_feat, histo_reconstruct_feat], dim=-1)
        fusion_feat = self.fc_fusion(fusion_feat).squeeze().unsqueeze(dim=0)

        # 报告的重建对齐
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
        original_histo_contrast_list = original_histo_contrast_list / original_histo_contrast_list.norm(dim=-1, keepdim=True)
        reconstruct_histo_contrast_list = reconstruct_histo_contrast_list / reconstruct_histo_contrast_list.norm(dim=-1, keepdim=True)

        logits_per_original = self.logit_scale1.exp() * original_histo_contrast_list @ reconstruct_histo_contrast_list.t()
        logits_original = logits_per_original[-1]
        original_probs = torch.softmax(logits_original, dim=-1)

        logits_per_reconstruct = logits_per_original.t()
        logits_reconstruct = logits_per_reconstruct[-1]
        reconstruct_probs = torch.softmax(logits_reconstruct, dim=-1)

        # update list
        self.original_histo_embedd_list[int(label)] = torch.cat([self.original_histo_embedd_list[int(label)][1:], histo], dim=0)
        self.reconstruct_histo_embedd_list[int(label)] = torch.cat([self.reconstruct_histo_embedd_list[int(label)][1:], histo_reconstruct_feat], dim=0)

        # ---->predict
        logits = self.classifier(fusion_feat.squeeze().unsqueeze(dim=0))  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        # 分子层面的语义对齐
        if label != self.n_classes - 1:  # 不是最后一个标签,防止越界
            fusion_contrast_list = torch.cat((self.fusion_embedd_list[0:label].detach(), self.fusion_embedd_list[label + 1:].detach()))
            sequence_contrast_list = torch.cat((self.sequence_embedd_list[0:label].detach(), self.sequence_embedd_list[label + 1:].detach()))
        else:
            fusion_contrast_list = self.fusion_embedd_list[0:label].detach()
            sequence_contrast_list = self.sequence_embedd_list[0:label].detach()

        fusion_contrast_list = fusion_contrast_list.view(-1, 512)
        fusion_contrast_list = torch.cat([fusion_contrast_list, fusion_feat])
        sequence_contrast_list = sequence_contrast_list.view(-1, 512)
        sequence_contrast_list =  torch.cat([sequence_contrast_list, sequence])

        #归一化
        fusion_contrast_list = fusion_contrast_list / fusion_contrast_list.norm(dim=-1, keepdim=True)
        sequence_contrast_list = sequence_contrast_list / sequence_contrast_list.norm(dim=-1, keepdim=True)

        logits_per_fusion = self.logit_scale2.exp() * fusion_contrast_list @ sequence_contrast_list.t()
        logits_fusion = logits_per_fusion[-1]
        fusion_probs = torch.softmax(logits_fusion, dim=-1)

        logits_per_sequence = logits_per_fusion.t()
        logits_sequence = logits_per_sequence[-1]
        sequence_probs = torch.softmax(logits_sequence, dim=-1)

        # update list
        self.fusion_embedd_list[int(label)] = torch.cat([self.fusion_embedd_list[int(label)][1:], fusion_feat], dim=0)
        self.sequence_embedd_list[int(label)] = torch.cat([self.sequence_embedd_list[int(label)][1:], sequence], dim=0)


        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'feature': fusion_feat
                          ,'fusion_probs': fusion_probs, 'sequence_probs': sequence_probs, 'length1':fusion_contrast_list.size()[0]
                          , 'original_probs': original_probs, 'reconstruct_probs': reconstruct_probs, 'length2': original_histo_contrast_list.size()[0]
                        }

        return results_dict
