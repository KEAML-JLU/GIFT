import os 
import torch
import random
import numpy as np
import torch.nn.functional as F
import json


def fetch_to_tensor(dicts, dict_type, device):
    return torch.tensor(dicts[dict_type], dtype=torch.float, device=device)
    

def fetch_to_sparse_tensor(dicts, dict_type, device):
    index = torch.from_numpy(np.vstack((dicts[dict_type].row, dicts[dict_type].col)).astype(np.int64))
    val = torch.from_numpy(dicts[dict_type].data)
    shape = dicts[dict_type].shape
    return torch.sparse.FloatTensor(index, val, shape).to(device)


def aggregate(adj_dict, input, input_one, other_type_num, softmax=False):
    aggregate_output = []
    aggregate_output_svd = []

    for i in range(other_type_num):
        adj = adj_dict[str(0) + str(i + 1)]
        adj_svd = adj_dict[str(0) + str(i + 1) + '_svd']
        
        aggregate_output_svd.append(torch.sparse.mm(adj_svd, input[i]))
        aggregate_output.append(torch.sparse.mm(adj, input[i]))
        #aggregate_output_svd.append((torch.sparse.mm(adj_svd, input[i])+torch.sparse.mm(adj_svd, input_one[i]))/2)
        #aggregate_output.append((torch.sparse.mm(adj, input[i])+torch.sparse.mm(adj, input_one[i]))/2)

    return aggregate_output, aggregate_output_svd


def sparse_dropout(mat, dropout):
    indices = mat.indices()
    values = F.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)


def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result


def learning_rate_decay(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']*0.98
        if lr > 0.0005:
            param_group['lr'] = lr
    return lr


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self,obj)

def save_res(params, acc, f1):
    from collections import defaultdict
    result=defaultdict(list)
    result[tuple([acc,f1])] = {
                                        'seed': params.seed,
                                        'weigh_dacay': params.weight_decay, 
                                        'lr': params.lr,
                                        'drop_out': params.drop_out,
                                        'ucl_temp': params.ucl_temp,
                                        'wcl_temp': params.wcl_temp,
                                        'hidden': params.hidden_size,
                                        'alpha': params.alpha,
                                        'beta': params.beta,
                                        'theta': params.theta,
                                        'svd': params.svd_q
                                        }
    os.makedirs(params.save_path, exist_ok=True)
    fname = params.save_name
    if not os.path.isfile(fname):
        with open(fname, mode='w') as f:
            f.write(json.dumps({str(k): result[k] for k in result}, cls=MyEncoder, indent=4))
            f.close()
    else:
        with open(fname, mode='a') as f:
            f.write(json.dumps({str(k): result[k] for k in result}, cls=MyEncoder, indent=4))
            f.close()