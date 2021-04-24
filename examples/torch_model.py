import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from graphmix.torch import SageConv, mp_matrix

class Net(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden):
        super(Net, self).__init__()
        self.conv1 = SageConv(dim_in, hidden, activation="relu", dropout=0.1)
        self.conv2 = SageConv(2*hidden, hidden, activation="relu", dropout=0.1)
        self.classifier = torch.nn.Linear(2*hidden, dim_out)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x, graph):
        edge_norm = mp_matrix(graph, x.device)
        x = self.conv1(x, edge_norm)
        x = self.conv2(x, edge_norm)
        x = F.normalize(x, p=2, dim=1)
        x = self.classifier(x)
        return x

    def loss(self, y_pred, y_true, mask):
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true.to(torch.float), reduction='none')
            loss = torch.mean(loss, axis=1)
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = F.cross_entropy(y_pred, y_true.flatten(), reduction='none')
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        return loss

    def metrics(self, y_pred, y_true, mask):
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            index = torch.where(mask)
            y_true = y_true[index]
            y_pred = torch.sigmoid(y_pred[index])
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
            true_pos = (y_pred + y_true == 2).sum()
            false_pos = (y_pred > y_true).sum()
            true_neg = (y_pred + y_true == 0).sum()
            false_neg = (y_pred < y_true).sum()
            accuracy = float(true_pos) / float(true_pos + false_pos)
            recall = float(true_pos) / float(true_pos + false_neg)
            f1_score = 2 * (accuracy * recall) / (accuracy + recall)
            return f1_score
        else:
            true_pos = y_pred.argmax(axis=1) == y_true.flatten()
            true_pos = int((true_pos * mask).sum())
            total = int(mask.sum())
            return true_pos / total

def torch_sync_data(*args):
    # all-reduce train stats
    t = torch.tensor(args, dtype=torch.float64, device='cuda')
    dist.barrier()  # synchronizes all processes
    dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t

def multi_label_eval(label, out):
    from sklearn import metrics
    out = torch.sigmoid(out)
    out[out > 0.5] = 1
    out[out <= 0.5] = 0
    label = label.cpu().detach().numpy()
    out = out.cpu().detach().numpy()
    micro_f1 = metrics.f1_score(label, out, average="micro")
    return micro_f1
