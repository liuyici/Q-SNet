"""
EEG Q-SNet for cross-subject emotion recognition 

"""

from einops import rearrange

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import numpy as np
import math
import random
import datetime
import time
import scipy.io
from modules import PseudoLabeledData, load_seed, load_seed_raw, fine_tuning_load_XY,load_seed_three_feature, load_seedv
from dataloader import *
from model.snn_layers import first_order_low_pass_layer, neuron_layer

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
# import lr_schedule
from   torch                            import autograd
from   torch.autograd                   import Variable
from   core_qnn.quaternion_layers       import *
import torchvision.transforms as transforms
import utils
from utils import LabelSmooth
import Adver_network
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r = input.norm(dim=1).detach()[0]
        cosine = F.linear(input, F.normalize(self.weight), r * torch.tanh(self.bias))
        output = cosine
        return output




class QLIF_Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.batchsize = args.batch_size
        self.sub = args.d_classes

        self.axon1 = first_order_low_pass_layer((args.dim_1,), args.length, self.batchsize, args.tau_m,
                                                True)
        self.snn1 = neuron_layer(args.dim_1, 256, args.length, self.batchsize, args.tau_m, True,
                                 True)

        self.axon2 = first_order_low_pass_layer((256,), args.length, self.batchsize, args.tau_m,
                                                True)
        self.snn2 = neuron_layer(256, 128, args.length, self.batchsize, args.tau_m, True,
                                 True)

        self.axon3 = first_order_low_pass_layer((128,), args.length, self.batchsize, args.tau_m, True)
        self.snn3 = neuron_layer(128, 100, args.length, self.batchsize, args.tau_m, True, False)

        self.dropout1 = torch.nn.Dropout(p=0.1, inplace=False)
        self.dropout2 = torch.nn.Dropout(p=0.1, inplace=False)
        self.linear = torch.nn.Linear(128 * args.window_size, args.n_classes)
        self.feat = ResidualAdd(nn.Sequential(
            QuaternionLinear(128, 128),
            nn.ELU(),
            # nn.Dropout(0.1),
            # GaussianNoise(1.0)
        ),
        downsample=nn.Linear(128, 128)  # 新增：调整残差分支维度
        )
        self.radius = 10
        self.fc2 = SLR_layer(256, args.n_classes)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        inputs = inputs[:, :, :-2]          # [32, 3, 308]
        inputs = inputs.view(inputs.size(0), -1)
        snn1_states = self.snn1.create_init_states()
        spike_l1, snn1_states = self.snn1(inputs, snn1_states)    # [B, 128, 12]
        drop_1 = self.dropout1(spike_l1)                              # dropout
        
        # axon2_states = self.axon2.create_init_states()
        snn2_states = self.snn2.create_init_states()
        spike_l2, snn2_states = self.snn2(drop_1, snn2_states)    # [B, 128, 12]
        spike_fea = spike_l2.reshape(spike_l2.size(0), -1)               # [B, 128*window_size]
        feat = self.feat(spike_fea)                                     # [B, 128]
        # out = self.fc2(feat)
        #out = self.linear(spike_fea)                                 # [B, n_classes]
        feat = self.radius * feat / (torch.norm(feat, dim=1, keepdim=True) + 1e-10) 
        return feat



class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = QuaternionLinearAutograd(308, 308, bias=False,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=True, quaternion_format=True, scale=False)
        self.queries = QuaternionLinear(308, 308)
        self.values = QuaternionLinear(308, 308)
        self.att_drop = nn.Dropout(dropout)
        self.projection = QuaternionLinear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        res_fea = x[:, :, -2:]    # [32, 3, 2]
        # print(res_fea.shape)
        q_x = x[:, :, :-2]          # [32, 3, 308]
        # print(q_x.shape)
        # q_x = self.layer1(q_x)
        # print(q_x.shape)
        x = torch.cat([q_x, res_fea], dim=-1)  # [32, 12, 310]
        queries = rearrange(self.queries(q_x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(q_x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(q_x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        
        out = rearrange(out, "b h n d -> b n (h d)")
        new_x = torch.cat([out, res_fea], dim=-1)
        out = self.projection(new_x)
        return out



class ResidualAdd(nn.Module):
    def __init__(self, fn, downsample=None):
        super().__init__()
        self.fn = fn
        self.downsample = downsample 

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)  
        if self.downsample is not None:
            res = self.downsample(res) 
        x += res
        return x



class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=14,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            ))
            )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, bottleneck_dim, n_classes):
        super().__init__()
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            SLR_layer(64, n_classes)
        )
        
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1).float()
        out = self.fc2(x)
        
        return x, out


class Discriminator(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            SLR_layer(64, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1).double()
        out = self.fc2(x)
        
        return out

# ! Rethink the use of Transformer for EEG signal
class Q_SNet(nn.Sequential):
    def __init__(self, emb_size=10, depth=6, bottleneck_dim=256, n_classes=4, **kwargs):
        super().__init__()
        self.encoder = TransformerEncoder(depth, emb_size)
        self.snn = QLIF_Model(args)
        self.head = ClassificationHead(emb_size, bottleneck_dim, n_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.snn(x)  

        return self.head(x)


class ExGAN():
    def __init__(self, args, nsub, fold):
        super(ExGAN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.batch_size = 128
        self.n_epochs = 50  #1000
        self.img_height = 22
        self.img_width = 600
        self.channels = 1
        self.c_dim = 4
        self.lr = 0.002
        self.lr2 = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.alpha = 0.0002
        self.dimension = (190, 50)
        self.nSub = nsub
        self.radius = 10
        self.start_epoch = 0
        self.root = '/home/lyc/research/seedv/ExtractedQuantFeatures/'

        self.pretrain = False

       

        self.img_shape = (self.channels, self.img_height, self.img_width)

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Q_SNet(emb_size=310, depth=6, bottleneck_dim=128, n_classes=5).float().cuda()
        self.domain_Discriminator = Discriminator(emb_size=10, n_classes=15).cuda()
        self.criterion = LabelSmooth(num_class=args.num_class).cuda()
        
    def schedule_lambda(self, epoch, total_epochs, max_lambda=0.2, k=2):

        p = epoch / max(1, (total_epochs - 1)) 
        return max_lambda * (2. / (1. + np.exp(-k * p)) - 1.)


    def get_source_data(self, feature="de_LDS"):
        if self.args.dataset == "seed":
            train_dataset, test_dataset, X, Y = load_seedv(self.args, self.args.file_path, session=self.args.session, feature=feature)
        return train_dataset, test_dataset, X, Y 
    
    def get_source_data_for_fine(self, X, Y):
        if self.args.dataset == "seed":
            dset_loaders = fine_tuning_load_XY(self.args, X, Y)
        return dset_loaders

    def test_suda(self, loader, model):
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.type(torch.FloatTensor).cuda()
                inputs = inputs.view(inputs.size(0), -1) 
                labels = labels
                _, outputs = model(inputs.float())
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        _, predictions = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])
    
        y_true = all_label.cpu().data.numpy()
        y_pred = predictions.cpu().data.numpy()
        labels = np.unique(y_true)
    
        ytest = label_binarize(y_true, classes=labels)
        ypreds = label_binarize(y_pred, classes=labels)
    
        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
        matrix = confusion_matrix(y_true, y_pred)
    
        return accuracy, f1, auc, matrix

    def nce_loss(self, features, labels, temperature=0.07):
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / temperature  # [N, N]
    
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().cuda() 
    
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).cuda()
        mask = mask * logits_mask
    
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)
    
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss



    def update_lr(self, optimizer, lr, iter_num):
        lr = lr * (1 + 0.01 * iter_num) ** (-0.75)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
    
    def _resample_match(self, X: torch.Tensor, target_len: int):
        """
        Randomly sample with replacement from X to length target_len
        X: [N, d]
        """
        if X.size(0) == target_len:
            return X
        idx = torch.randint(0, X.size(0), (target_len,), device=X.device)
        return X[idx]

    def _to_tensor(self, x, device, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=dtype)
        return x

    def train(self, fold):

        train_dataset, test_dataset, X, Y  = self.get_source_data(feature="de_LDS")

        # self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.domain_Discriminator.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.005)
        self.optimizer = torch.optim.SGD(
                         list(self.model.parameters()) + list(self.domain_Discriminator.parameters()),  # 同时优化model和domain_Discriminator的参数
                         lr=self.lr,
                         momentum=0.9,
                         weight_decay=0.005
                        )
        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0
        epochs_acc = []
        B = self.args.batch_size
        for e in range(self.n_epochs):
            # self.optimizer = self.update_lr(self.optimizer, self.lr, e)
            # print('The learning rate is:', self.optimizer.param_groups[0]['lr'])
            self.model.train()
            for i, data in enumerate(train_dataset):
                x_src = list()
                y_src = list()
                d_src = list()
                index = 0
                for domain_idx in range(16 - 1):
                    tmp_x = data['Sx' + str(domain_idx + 1)].float().cuda()
                    tmp_y = data['Sy' + str(domain_idx + 1)].long().cuda()
                    labels = torch.from_numpy(np.array([[index] * args.batch_size]).T).type(torch.FloatTensor).flatten().long().cuda()
                    x_src.append(tmp_x)
                    d_src.append(labels)
                    y_src.append(tmp_y)
                    index += 1
                x_trg = data['Tx'].float().cuda()
                test_label = data['Ty'].long().cuda()
                
                img = torch.cat(x_src, dim=0)
               
                label = torch.cat(y_src, dim=0)
                domain_label = torch.cat(d_src, dim=0)
                img = img.view(img.size(0),  -1) 
                # img = img.permute(0, 2, 1)  # [100, 62, 5, 3]
                x_trg = x_trg.view(x_trg.size(0), -1) 
                # x_trg = x_trg.permute(0, 2, 1)  # [100, 62, 5, 3]
                 #img.view(img.size(0), img.size(1), -1)
                tok, outputs = self.model(img.float())
                tok_target, outputs_target = self.model(x_trg.float())
                pre_target = torch.nn.functional.softmax(outputs_target,dim = 1)
                
                mmd_b_vals, mmd_t_vals = [], []
                for d in range(15):
                    s_slice = slice(d*B, (d+1)*B)
                    src_tok_d    = tok[s_slice]                       # [B, 64] features for MMD space -> we should use the feature before classifier
                    src_lab_d    = label[s_slice].reshape(B, 1)       # [B, 1]
                    tgt_tok_eq   = self._resample_match(tok_target, B)    # [B, 64]
                    tgt_prob_eq  = self._resample_match(pre_target, B)    # [B, C]

                    mb = utils.marginal(src_tok_d, tgt_tok_eq)
                    mt = utils.conditional(
                            src_tok_d,
                            tgt_tok_eq,
                            src_lab_d,
                            tgt_prob_eq,
                            0.5,
                            5,
                            None
                    )
                    mmd_b_vals.append(self._to_tensor(mb, outputs.device))
                    mmd_t_vals.append(self._to_tensor(mt, outputs.device))

                mmd_b_loss = torch.stack(mmd_b_vals).mean()
                mmd_t_loss = torch.stack(mmd_t_vals).mean()
                
                lambda_adv = self.schedule_lambda(e, self.n_epochs)
                features_s_Adver = Adver_network.ReverseLayerF.apply(tok, lambda_adv)
                outputs_D = self.domain_Discriminator(features_s_Adver)
                Adver_domain_labels_loss = self.criterion(outputs_D, domain_label.flatten())

                if isinstance(mmd_b_loss, np.ndarray):
                     mmd_b_loss = torch.tensor(mmd_b_loss, device=outputs.device, dtype=torch.float32)

                if isinstance(mmd_t_loss, np.ndarray):
                    mmd_t_loss = torch.tensor(mmd_t_loss, device=outputs.device, dtype=torch.float32)
                if isinstance(Adver_domain_labels_loss, np.ndarray):
                    Adver_domain_labels_loss = torch.tensor(Adver_domain_labels_loss, device=outputs.device, dtype=torch.float32)
                MMD_loss = mmd_b_loss + mmd_t_loss
                slc_loss = self.criterion(outputs, label)
                # lambda_mmd = 0.5 
                # lambda_adv = self.schedule_lambda(e, self.n_epochs) 
                if e<20:
                    lambda_mmd = 0.1
                else:
                    lambda_mmd = 1
                loss = slc_loss + lambda_mmd*MMD_loss + Adver_domain_labels_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            out_epoch = time.time()

            if (e + 1) % 1 == 0:
                start_test = True
                with torch.no_grad():        
                    self.model.eval()
        
                    for batch_idx, tar_data in enumerate(test_dataset):
                        Tx = tar_data['Tx']
                        Ty = tar_data['Ty']
                        Tx = Tx.float().cuda()
                        Tx = Tx.view(Tx.size(0), -1) 
                        # Tx = Tx.permute(0, 2, 1) 
                        Tok, Cls = self.model(Tx)
                        if start_test:
                            all_output = Cls.float().cpu()
                            all_label = Ty.float()
                            start_test = False
                        else:
                            all_output = torch.cat((all_output, Cls.float().cpu()), 0)
                            all_label = torch.cat((all_label, Ty.float()), 0)
                        loss_test = self.criterion_cls(Cls.float().cpu(), Ty.long())
                torch.cuda.empty_cache()  
                y_pred = torch.max(all_output, 1)[1]
                acc = float((y_pred == all_label).cpu().numpy().astype(int).sum()) / float(all_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                epochs_acc.append(acc)
                # print('The epoch is:', e, '  The accuracy is:', acc)
                print('Epoch:', e,
                      '  Train loss: %.4f' % loss.item(),
                      '  cls: %.4f' % slc_loss.detach().cpu().numpy(),
                      '  MMD: %.4f' % MMD_loss.item(),
                      '  adv: %.4f' % Adver_domain_labels_loss.detach().cpu().numpy(),
                      '  lambda_adv: %.4f' % lambda_adv,
                      '  Train acc: %.4f' % train_acc,
                      '  Test acc: %.4f' % acc)
             
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = Ty
                    Y_pred = y_pred

        averAcc = averAcc / num
        print('The average accuracy of n_epochs%d is:' %(e+1), averAcc)
        print('The best accuracy of n_epochs%d is:' %(e+1), bestAcc)
     
        return bestAcc, averAcc, Y_true, Y_pred, X, Y, self.model, epochs_acc


    def fine_tuning(self, args, X, Y, model):
        dset_loaders = self.get_source_data_for_fine(X, Y)
        parameter_model = model.parameters()
        self.optimizer = torch.optim.Adam(parameter_model, lr=self.lr2, betas=(self.b1, self.b2))
        len_train_source = len(dset_loaders["source"])
        len_train_target = len(dset_loaders["target"])
    
        best_acc = 0.0
        final_acc = 0
        final_f1 = 0
        final_auc = 0
        final_mat = []
        for i in range(args.max_iter2):
            if i % 1 == 0:
                with torch.no_grad():      
                    model.eval()
                    best_acc, best_f1, best_auc, best_mat = self.test_suda(dset_loaders, model)
                    if final_acc < best_acc:
                        final_acc = best_acc
                        final_f1 = best_f1
                        final_auc = best_auc
                        final_mat = best_mat  
                    if i == 0:
                        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(i, best_acc, best_f1, best_auc)
                    else: 
                        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f} \t loss: {:.4f}".format(i, best_acc, best_f1, best_auc, total_loss.item())
                    print(log_str)
            model.train()
            if i % len_train_source == 0:
                iter_source = iter(dset_loaders["source"])
            if i % len_train_target == 0:
                iter_target = iter(dset_loaders["target"])
           
            inputs_source_, labels_source = next(iter_source)
            inputs_target_, ture_labels_target = next(iter_target)
            inputs_source_ = inputs_source_.type(torch.FloatTensor)
            labels_source = labels_source.type(torch.LongTensor)
            inputs_target_ = inputs_target_.type(torch.FloatTensor)
            ture_labels_target = ture_labels_target.type(torch.LongTensor)
            inputs_source, labels_source = inputs_source_.cuda(), labels_source.cuda()
            inputs_target, ture_labels_target = inputs_target_.cuda(), ture_labels_target.cuda()
            inputs_source = inputs_source.view(inputs_source.size(0), -1) 
            # inputs_source = inputs_source.permute(0, 2, 1) 
            inputs_target = inputs_target.view(inputs_target.size(0), -1)  
            # inputs_target = inputs_target.permute(0, 2, 1) 
            features_source, outputs_source = model(inputs_source)
            features_target, outputs_target = model(inputs_target)
            classifier_loss = self.criterion_cls(outputs_source, labels_source.flatten())
            pre_target = F.softmax(outputs_target, dim=1)
            all_features = torch.cat([features_source, features_target], dim=0)
            all_labels = torch.cat([labels_source, torch.argmax(pre_target, dim=1)], dim=0)
            nce = self.nce_loss(all_features, all_labels, temperature=0.07)
            
            # ce_loss = torch.mean(utils.Entropy(pre_target))
            # mmd_b_loss = utils.marginal(features_source,features_target)
            # mmd_t_loss = utils.conditional(
            #            features_source,
            #            features_target,
            #            labels_source.reshape((200, 1)),
            #            torch.nn.functional.softmax(outputs_target,dim = 1),
            #            2.0,
            #            5,
            #            None)
            # mmd_loss = 0.5*mmd_b_loss + 0.5*mmd_t_loss
            CORAL = utils.CORAL_loss(outputs_source, outputs_target)
            total_loss = classifier_loss   + 2 * CORAL # + mmd_loss
    
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return final_acc, final_f1, final_auc, final_mat, model

def main(args):
    pre_train = []
    tuning = []
    time_costs = []
    acc = []
    f1 = []
    mat = []
    auc = []
    result_write = open("/home/lyc/research/research_7/seed-v-9376/snapshot.txt", "w")
    for i in range(16):
        args.target = 16 - i
        # starttime = datetime.datetime.now()
        seed_n = 1

        result_write.write('--------------------------------------------------')
        # print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        print('Subject %d' % (i+1))
        
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        ba = 0
        aa = 0
        pre_train_Acc = 0
        averAcc = 0

        exgan = ExGAN(args, i + 1, 1)
        start_time = time.time()
        ba, aa, _, _, X, Y, model, epochs_acc  = exgan.train(1)
        end_time = time.time()
        time_cost = end_time - start_time
        time_costs.append(time_cost)
        print(f"trian time: {time_cost:.4f} 秒")
        num_params = count_parameters(model)
        print(f"total num_params: {num_params}")
        final_acc, final_f1, final_auc, final_mat, model = exgan.fine_tuning(args, X, Y, model)

        result_write.write('pre_training acc is:' + str(ba) + "\n")
        result_write.write('fine_tuning acc is:' + str(final_acc) + "\n")
        pre_train_Acc = ba
        tuning_Acc = final_acc
        # plot_confusion_matrix(Y_true, Y_pred, i+1)
        pre_train.append(pre_train_Acc)
        tuning.append(tuning_Acc)
        f1.append(final_f1)
        auc.append(final_auc)
        # endtime = datetime.datetime.now()
        # print('subject %d duration: '%(i+1) + str(endtime - starttime))
        print('pre_training acc is:', pre_train)
        print('fine_tuning acc is:', tuning)
    mean_auc = sum(auc)/len(auc)
    result_std_auc = np.std(auc)
    mean_f1 = sum(f1)/len(f1)
    result_std_f1 = np.std(f1)
    pre_ave = sum(pre_train) / len(pre_train)
    tuning_ave = sum(tuning) / len(tuning)
    result_std = np.std(tuning)
    print('------------------------pre-training result--------------------------', pre_train)
    print('------------------------fin-tuning result--------------------------', tuning)
    print('------------------------pre-training average result--------------------------', pre_ave)
    print('------------------------fin-tuning average result--------------------------', tuning_ave)
    print('------------------------fin-tuning std result--------------------------', result_std)
    print('------------------------fin-tuning mean f1 score--------------------------', mean_f1)
    print('------------------------fin-tuning std f1 score--------------------------', result_std_f1)
    print('------------------------fin-tuning mean auc--------------------------', mean_auc)
    print('------------------------fin-tuning std auc--------------------------', result_std_auc)
    result_write.write('--------------------------------------------------')
    result_write.write(f"All accuracy is: {pre_train}\n")
    result_write.write(f"All subject Aver accuracy is: {tuning}\n")
    result_write.write(f"All subject Mean F1 score is: {mean_f1}\n")
    result_write.write(f"All subject Mean AUC is: {mean_auc}\n")
    result_write.write(f"All subject F1 std is: {result_std_f1}\n")
    result_write.write(f"All subject AUC std is: {result_std_auc}\n")
    result_write.close()
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spherical Space Domain Adaptation with Pseudo-label Loss')
    parser.add_argument('--baseline', type=str, default='MSTN', choices=['MSTN', 'DANN'])
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--dataset',type=str,default='seed')
    parser.add_argument('--source', type=str, default='amazon')
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=1, help="Iteration repetitions")
    parser.add_argument('--test_interval', type=int, default=1, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--mixed_sessions', type=str, default='per_session', help="[per_session | mixed]")
    parser.add_argument('--lr_a', type=float, default=0.1, help="learning rate 1")
    parser.add_argument('--lr_b', type=float, default=0.1, help="learning rate 2")
    parser.add_argument('--radius', type=float, default=10, help="radius")
    parser.add_argument('--num_class',type=int,default=5,help='the number of classes')
    parser.add_argument('--stages', type=int, default=1, help='the number of alternative iteration stages')
    parser.add_argument('--max_iter1',type=int,default=100)
    parser.add_argument('--max_iter2', type=int, default=1000)
    parser.add_argument('--batch_size',type=int,default=50)
    parser.add_argument('--batch_size_fine',type=int,default=200)
    parser.add_argument('--seed', type=int, default=123, help="random seed number ")
    parser.add_argument('--hidden_size', type=int, default=256, help="Bottleneck (features) dimensionality")
    parser.add_argument('--bottleneck_dim', type=int, default=128, help="Bottleneck (features) dimensionality")
    parser.add_argument('--session', type=int, default=1, help="random seed number ")
    parser.add_argument('--gamma', type=int, default=1, help="gamma for Adver_network ")
    parser.add_argument('--file_path', type=str, default='/home/lyc/research/seedv/ExtractedQuantFeatures/', help="Path from the current dataset")
    parser.add_argument('--log_file')
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--d_classes', type=int, default=15)
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--tau_m', type=int, default=1)
    parser.add_argument('--train_coefficients', type=int, default=True)
    parser.add_argument('--train_bias', type=int, default=True)
    parser.add_argument('--membrane_filter', type=int, default=False)
    parser.add_argument('--dim_1', type=int, default=308)
    parser.add_argument('--length', type=int, default=25)
    #####
    parser.add_argument('--ila_switch_iter', type=int, default=1, help="number of iterations when only DA loss works and sim doesn't")
    parser.add_argument('--n_samples', type=int, default=2, help='number of samples from each src class')
    parser.add_argument('--mu', type=int, default=80, help="these many target samples are used finally, eg. 2/3 of batch")  # mu in number
    parser.add_argument('--k', type=int, default=3, help="k")
    parser.add_argument('--msc_coeff', type=float, default=1.0, help="coeff for similarity loss")
    #####
    args = parser.parse_args()
    main(args)
