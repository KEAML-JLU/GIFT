import numpy as np
import pickle as pkl
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import math
from scipy import sparse
from sklearn import metrics
from utils import fetch_to_tensor, fetch_to_sparse_tensor, learning_rate_decay
from model import GIFT
from CL import Classifier, UCL, WCL
from kmeans import ConstrainedSeedKMeans


class Trainer(object):
    def __init__(self, params):
        self.params = params
        self.dataset_name = params.dataset
        self.max_epoch = params.max_epoch
        self.save_path = params.save_path
        self.device = params.device
        self.hidden_size = params.hidden_size
        self.lr = params.lr
        self.weight_decay = params.weight_decay
        self.concat_word_emb = params.concat_word_emb
        self.type_names = params.type_num_node
        self.data_path = params.data_path
        self.dropout = params.drop_out
        
        self.ucl_temp = params.ucl_temp
        self.wcl_temp = params.wcl_temp
        self.alpha = params.alpha
        self.beta = params.beta
        self.theta = params.theta
        self.svd_q = params.svd_q

        self.adj_dict, self.features_dict, self.train_idx, self.valid_idx, self.test_idx, self.labels, self.nums_node = self.load_data()
        
        
        self.label_num = len(set(self.labels))
        self.labels = torch.tensor(self.labels).to(self.device)
        self.pseudo_label = torch.LongTensor(self.labels.shape).fill_(-1).to(self.device)
        self.out_features_dim = [self.label_num, self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size]
        in_fea_final = self.out_features_dim[1] + self.out_features_dim[2] + self.out_features_dim[3]
        self.in_features_dim = [0, self.nums_node[1], self.nums_node[2], self.nums_node[-1], in_fea_final]

        if self.concat_word_emb: self.in_features_dim[-1] += self.features_dict['word_emb'].shape[-1]
        self.model = GIFT(self.adj_dict, self.features_dict, self.in_features_dim, self.out_features_dim, params)
        self.model = self.model.to(self.device)
        
        self.cluster = ConstrainedSeedKMeans(n_clusters=self.label_num, n_init=10, verbose=False)
        self.cls = Classifier(self.in_features_dim[-1], self.out_features_dim[-1], self.out_features_dim[0], self.dropout)
        self.cls = self.cls.to(self.device)
        self.ucl = UCL(self.in_features_dim[-1], self.hidden_size, self.ucl_temp)
        self.ucl = self.ucl.to(self.device)
        self.wcl = WCL(self.in_features_dim[-1], self.hidden_size, self.wcl_temp)
        self.wcl = self.wcl.to(self.device)
        
        
        self.optim = optim.Adam([{'params': self.model.parameters()},
                         {'params': self.cls.parameters()},
                         {'params': self.ucl.parameters()},
                         {'params': self.wcl.parameters()},    
                        ], lr=self.lr, weight_decay=self.weight_decay)


    def train(self):
        global_best_acc = 0
        global_best_f1 = 0
        global_best_epoch = 0
        best_test_acc = 0
        best_test_f1 = 0
        best_valid_epoch = 0
        best_valid_f1 = 0
        best_valid_acc = 0
        acc_valid = 0
        loss_valid = 0
        f1_valid = 0
        acc_test = 0
        loss_test = 0
        f1_test = 0
        best_acc = 0
        best_f1 = 0
        for i in range(1, self.max_epoch + 1):
            
            t=time.time()
            doc_fea, doc_fea_svd = self.model(i)
            
            output, output_svd = self.cls(doc_fea), self.cls(doc_fea_svd)
            
            ori_train_scores = output[self.train_idx]
            aug_train_scores = output_svd[self.train_idx]
            
            train_labels = self.labels[self.train_idx]
            self.pseudo_label[self.train_idx] = train_labels 
            # compute the prototype
            doc_fea_clone = doc_fea.clone().detach()
            self.cluster.fit(doc_fea_clone, self.pseudo_label)
            prototype = self.cluster.cluster_centers_
            weak_label = self.cluster.indices
            
            cls_loss = (F.cross_entropy(ori_train_scores, train_labels) + F.cross_entropy(aug_train_scores, train_labels)) * self.theta
            ucl_loss = self.ucl(doc_fea, doc_fea_svd)

            wcl_loss = self.wcl(doc_fea, doc_fea_svd, prototype, weak_label)
            loss = cls_loss + self.alpha * ucl_loss + self.beta * wcl_loss

            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            torch.cuda.empty_cache()
            
            loss = loss.item()
            
            ori_acc = torch.eq(torch.argmax(ori_train_scores, dim=-1), train_labels).float().mean().item()
            aug_acc = torch.eq(torch.argmax(aug_train_scores, dim=-1), train_labels).float().mean().item()
            acc = (ori_acc+aug_acc)/2

            if i%5 == 0:

                acc_test, loss_test, f1_test = self.test(i)
                if acc_test > global_best_acc:
                    global_best_acc = acc_test
                    global_best_f1 = f1_test
                    global_best_epoch = i

                best_acc = global_best_acc
                best_f1 = global_best_f1
                best_epoch = global_best_epoch
                
            if i%100==0:
                print('GLOBAL: TEST ACC', global_best_acc, 'TEST F1', global_best_f1, 'EPOCH', global_best_epoch)
        return best_acc, best_f1

    def test(self, epoch):
        t = time.time()
        self.model.training = False
        self.cls.training = False
        self.ucl.training = False
        self.wcl.training = False
        doc_fea, _ = self.model(0)
        output = self.cls(doc_fea)

        with torch.no_grad():
            test_scores = output[self.test_idx]
            test_labels = self.labels[self.test_idx]
            loss_test = F.cross_entropy(test_scores, test_labels).item()
            acc_test = torch.eq(torch.argmax(test_scores, dim=-1), test_labels).float().mean().item()
            f1_test = metrics.f1_score(test_labels.detach().cpu().numpy(),torch.argmax(test_scores,-1).detach().cpu().numpy(),average='macro')
            print('Test  loss: {:.4f} acc: {:.4f} f1: {:.4f} time: {:.4f}'.format(loss_test, acc_test, f1_test, time.time() - t))
        self.model.training = True
        self.cls.training = True
        self.ucl.training = True
        self.wcl.training = True
        return acc_test, loss_test, f1_test   

    def load_data(self):
        start = time.time()
        adj_dict = {}
        feature_dict = {}
        nums_node = []
        for i in range(1, len(self.type_names)):
            adj_dict[str(0) + str(i)] = pkl.load(
                    open(self.data_path + './adj_{}2{}.pkl'.format(self.type_names[0], self.type_names[i]), 'rb'))
            if i == 1:
                nums_node.append(adj_dict[str(0) + str(i)].shape[0])
            if i != 3:
                adj_dict[str(i) + str(i)] = pkl.load(
                    open(self.data_path + './adj_{}.pkl'.format(self.type_names[i]), 'rb'))
                nums_node.append(adj_dict[str(i) + str(i)].shape[0])
            if i == 3:
                feature_dict[str(i)] = pkl.load(
                    open(self.data_path + './{}_emb.pkl'.format(self.type_names[i]), 'rb'))
                nums_node.append(feature_dict[str(i)].shape[0])
                nums_node.append(feature_dict[str(i)].shape[1])
            else:
                feature_dict[str(i)] = sparse.coo_matrix(np.eye(nums_node[i], dtype=np.float32))

        feature_dict['word_emb'] = torch.tensor(pkl.load(
            open(self.data_path + './word_emb.pkl', 'rb')), dtype=torch.float).to(self.device)
        ent_emb=feature_dict['3']
        ent_emb_normed = ent_emb / np.sqrt(np.square(ent_emb).sum(-1, keepdims=True))
        adj_dict['01'] = sparse.coo_matrix(adj_dict['01'], dtype=np.float32)
        adj_dict['11'] = sparse.coo_matrix(adj_dict['11'], dtype=np.float32)
        adj_dict['33'] = np.matmul(ent_emb_normed, ent_emb_normed.transpose())
        adj_dict['33'] = adj_dict['33'] * np.float32(adj_dict['33'] > 0)
        adj_dict['33'] = sparse.coo_matrix(adj_dict['33'], dtype=np.float32)
        adj_dict['22'] = adj_dict['22'].astype(np.float32)
        adj_dict['02'] = adj_dict['02'].astype(np.float32)
        adj_dict['03'] = adj_dict['03'].astype(np.float32)
        
        adj = {}
        feature = {}
        for i in adj_dict.keys():
            adj[i] = fetch_to_sparse_tensor(adj_dict, i, self.device)
            if i in ['01', '02', '03']:
                print('Performing adj_{} SVD...'.format(i))
                svd_u,s,svd_v = torch.svd_lowrank(adj[i], q=self.svd_q)
                adj[i+'_svd'] = svd_u @ torch.diag(s) @ svd_v.T
                del svd_u, s, svd_v
                print('SVD done.')
            
        for i in feature_dict.keys():
            if i == "3" or i == "word_emb":
                feature[i] = fetch_to_tensor(feature_dict, i, self.device)
            else:
                feature[i] = fetch_to_sparse_tensor(feature_dict, i, self.device)
        
        train_set = json.load(open(self.data_path + './train_idx.json'))
        test_set = json.load(open(self.data_path + './test_idx.json'))
        labels = json.load(open(self.data_path + './labels.json'))
        
        data_index = train_set + test_set
        Sumofquery = len(data_index)
        print(Sumofquery)
        label_dict = {}
        for i in set(labels):
            label_dict[i] = []
        for j, label in enumerate(labels):
            label_dict[label].append(j)
        len_train_idx = len(label_dict) * 20 
        train_list = []
        valid_list = []
        byclass = True if self.dataset_name=='ohsumed' else False 
        if not byclass:
            for i in label_dict.items():
                train_list.append(20)
        else:
            ratio = len_train_idx / Sumofquery
            print("train_ratio: ", ratio)
            residue = []
            max_supp = len_train_idx
            for val in label_dict.values():
                num = math.modf(len(val) * ratio)
                residue.append(num[0])
                train_list.append(int(num[1]))
                max_supp -= int(num[1])
            sorted_list = sorted(range(len(residue)), key= lambda i : residue[i], reverse=True)[ : max_supp]
            for i, val in enumerate(train_list):
                if i in sorted_list:
                    train_list[i] += 1
        valid_list = train_list
        train_idx = []
        valid_idx = []
        test_idx = []
        for i, idxs in enumerate(label_dict.values()):
            np.random.shuffle(idxs)
            for j, idx in enumerate(idxs):
                if j < train_list[i]:
                    train_idx.append(idx)
                elif j >= train_list[i] and j < train_list[i] + valid_list[i]:
                    valid_idx.append(idx)
                else:
                    test_idx.append(idx)
        print(len(train_idx))
        print(len(valid_idx))
        print(len(test_idx)) 
        print('data process time: {}'.format(time.time()-start))
        return adj, feature, train_idx, valid_idx, test_idx, labels, nums_node

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.model.state_dict(), path + './{}/save_model_new'.format(self.dataset_name))

    def load(self):
        self.model.load_state_dict(torch.load(self.save_path))
