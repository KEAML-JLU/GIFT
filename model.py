import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import aggregate
from GCN import GCN

class GIFT(nn.Module):
    def __init__(self, adj_dict, features_dict, in_features_dim, out_features_dim, params):
        super(GIFT, self).__init__()
        self.adj = adj_dict
        self.feature = features_dict
        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.type_num = len(params.type_num_node)
        self.drop_out = params.drop_out
        self.concat_word_emb = params.concat_word_emb
        self.device = params.device
        self.GCNs = nn.ModuleList()
        self.GCNs_2 = nn.ModuleList()

        for i in range(1, self.type_num):
            self.GCNs.append(GCN(self.in_features_dim[i], self.out_features_dim[i]).to(self.device))
            self.GCNs_2.append(GCN(self.out_features_dim[i], self.out_features_dim[i]).to(self.device))

    def embed_component(self, norm=True):
        output = []
        output_one = []
        for i in range(self.type_num - 1):
            if i == 1 and self.concat_word_emb:
                l1 = self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)], identity=True)
                l2 = self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)], l1)
                temp_emb = torch.cat([F.dropout(l2, p=self.drop_out, training=self.training), self.feature['word_emb']], dim=-1)
                output.append(temp_emb)
                output_one.append(torch.cat([F.dropout(l1, p=self.drop_out), self.feature['word_emb']], dim=-1))
            elif i == 0:
                l1 = self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)], identity=True)
                l2 = self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)], l1)
                temp_emb = F.dropout(l2, p=self.drop_out, training=self.training)
                output.append(temp_emb)
                output_one.append(F.dropout(l1, p=self.drop_out))
            else:
                l1 = self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)])
                l2 = self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)], l1)
                temp_emb = F.dropout(l2, p=self.drop_out, training=self.training)
                output.append(temp_emb)
                output_one.append(F.dropout(l1, p=self.drop_out))
        refined_text_input, refined_text_input_svd = aggregate(self.adj, output, output_one, self.type_num - 1) 
        if norm:
            refined_text_input_normed = []
            refined_text_input_normed_svd = []
            for i in range(self.type_num - 1):
                refined_text_input_normed.append(refined_text_input[i] / (refined_text_input[i].norm(p=2, dim=-1,keepdim=True) + 1e-9))
                refined_text_input_normed_svd.append(refined_text_input_svd[i] / (refined_text_input_svd[i].norm(p=2, dim=-1,keepdim=True) + 1e-9))
        else:
            refined_text_input_normed = refined_text_input
            refined_text_input_normed_svd = refined_text_input_svd
        return refined_text_input_normed, refined_text_input_normed_svd
    
    def forward(self, epoch):
        refined_text_input_normed, refined_text_input_normed_svd = self.embed_component()
        
        Doc_features = torch.cat(refined_text_input_normed, dim=-1)
        Doc_features_svd = torch.cat(refined_text_input_normed_svd, dim=-1)
        

        return Doc_features, Doc_features_svd
