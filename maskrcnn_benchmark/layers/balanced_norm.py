import torch
from torch import nn
from maskrcnn_benchmark.modeling.utils import cat
import torch.nn.functional as F
import numpy as np
class LearnableBalancedNorm1d(nn.Module):
    """
    LearnableBalancedNorm1d.
    """

    def __init__(self, num_features, eps=1e-5, normalized_probs=False):
        super(LearnableBalancedNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.normalized_probs = normalized_probs
        self.labeling_prob_theta = nn.Parameter(torch.randn(50))

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))

    def forward(self, relation_logits, rel_labels):
        relation_logits = cat(relation_logits) if isinstance(relation_logits, tuple) else relation_logits
        self._check_input_dim(relation_logits)

        labeling_prob = torch.sigmoid(self.labeling_prob_theta)
        labeling_prob = torch.cat((torch.ones(1).cuda(), labeling_prob)) + self.eps

        relation_probs_norm = F.softmax(relation_logits, dim=-1) / labeling_prob
        if self.normalized_probs:
            # relation_probs_norm /= (torch.sum(relation_probs_norm, dim=-1).view(-1, 1) + self.eps)
            # relation_probs_norm = F.softmax(relation_probs_norm, dim=-1)
            # import pdb; pdb.set_trace()
            relation_probs_norm[:, 0] = 1 - relation_probs_norm[:, 1:].sum(1)

        return relation_probs_norm, labeling_prob

class BalancedNorm1d(nn.Module):
    """
    BalancedNorm1d.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, init_prob=0.03, track_running_stats=True, normalized_probs=True, with_gradient=False):
        super(BalancedNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.normalized_probs = normalized_probs
        self.with_gradient = with_gradient
        self.init_prob2 = 1e-6
        if self.track_running_stats:
            self.init_prob = init_prob
            self.register_buffer("running_labeling_prob", torch.tensor([init_prob] * num_features))
            self.register_buffer("running_column_prob", torch.eye(num_features, num_features))
            self.register_buffer("running_row_prob", torch.eye(num_features, num_features))
            self.register_buffer("running_label", torch.ones(num_features))
            self.running_labeling_prob[0] = 1 # BG labeling prob is always one
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_labeling_prob", None)
            self.register_parameter("running_column_prob", None)
            self.register_parameter("running_row_prob", None)
            self.register_parameter('num_batches_tracked', None)
            self.register_parameter("running_label", None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_labeling_prob.fill_(self.init_prob)
            self.running_labeling_prob[0] = 1 # BG labeling prob is always one
            self.running_column_prob = torch.eye(self.running_column_prob.shape[0], self.running_column_prob.shape[1])
            self.running_row_prob = torch.eye(self.running_row_prob.shape[0], self.running_row_prob.shape[1])
            self.running_label = torch.ones(self.running_label.shape[0])
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        # if self.affine:
        #     init.ones_(self.weight)
        #     init.zeros_(self.bias)

    # def forward(self, x):
    #     # Cast all fixed parameters to half() if necessary
    #     if x.dtype == torch.float16:
    #         self.weight = self.weight.half()
    #         self.bias = self.bias.half()
    #         self.running_mean = self.running_mean.half()
    #         self.running_var = self.running_var.half()

    #     scale = self.weight * self.running_var.rsqrt()
    #     bias = self.bias - self.running_mean * scale
    #     scale = scale.reshape(1, -1, 1, 1)
    #     bias = bias.reshape(1, -1, 1, 1)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))
    def resistance(self,weight):
        return torch.log(weight/torch.sum(weight) + 0.0001)
    def forward(self, relation_logits, rel_labels):
        '''
        Takes in the same parameters as those of common loss functions.
        Parameters:
        - input: Input probability score (logits passed through Softmax).
        - target: Target 
        '''
        # import pdb; pdb.set_trace()
        relation_logits = cat(relation_logits) if isinstance(relation_logits, tuple) else relation_logits
        rel_labels = cat(rel_labels) if isinstance(rel_labels, list) else rel_labels
        self._check_input_dim(relation_logits)

        exponential_average_factor = 0.0
        exponential_average_factor_column = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    exponential_average_factor_column = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
                    exponential_average_factor_column = self.momentum


        if self.training:
            # import pdb; pdb.set_trace()
            fg_idxs = (rel_labels != 0)
            fg_relation_probs = F.softmax(relation_logits[fg_idxs], dim=-1)

            if fg_relation_probs.shape[0] != 0:
                    # confuse  51*51
                    pred_labels = torch.argmax(relation_logits[:,1:],-1) + 1
                    pred_labels_one_hot = torch.zeros( fg_relation_probs.shape[1],fg_relation_probs.shape[1]).unsqueeze(0).repeat(fg_relation_probs.shape[0],1,1)
                    pred_labels_one_hot[list(range(len(fg_relation_probs))), rel_labels[fg_idxs], pred_labels[fg_idxs]] = 1.
                    pred_labels_one_hot = torch.sum(pred_labels_one_hot,0)
                    fg_pred_relation_probs = pred_labels_one_hot.to(relation_logits.device)
                    # logit  51
                    pred_labels_one_hot = torch.zeros_like(fg_relation_probs, dtype=torch.int)
                    pred_labels_one_hot[list(range(len(fg_relation_probs))), rel_labels[fg_idxs]] = 1
                    labeling_prob = torch.sum(fg_relation_probs * pred_labels_one_hot, dim=0) / torch.sum(pred_labels_one_hot, dim=0)
                    non_nan_idxs = ~torch.isnan(labeling_prob)
                    # relation ratio  51
                    rel_labels_one_hot = torch.zeros_like(fg_relation_probs, dtype=torch.int)
                    rel_labels_one_hot[list(range(len(fg_relation_probs))), rel_labels[fg_idxs]] = 1




            if self.with_gradient:
                if fg_relation_probs.shape[0] != 0:
                        self.running_column_prob  =  fg_pred_relation_probs + self.running_column_prob
                self.running_label += torch.sum(rel_labels_one_hot, dim=0)
                self.running_labeling_prob[non_nan_idxs] = exponential_average_factor * labeling_prob[non_nan_idxs] + (1 - exponential_average_factor) * self.running_labeling_prob[non_nan_idxs]
                

            else:
                with torch.no_grad():
                    if fg_relation_probs.shape[0] != 0:
                            self.running_column_prob  =  fg_pred_relation_probs + self.running_column_prob
                    self.running_label += torch.sum(rel_labels_one_hot, dim=0)
                    self.running_labeling_prob[non_nan_idxs] = exponential_average_factor * labeling_prob[non_nan_idxs] + (1 - exponential_average_factor) * self.running_labeling_prob[non_nan_idxs]
        
        beta = 1.
        beta2 = 0.
        running_column_prob = (torch.diag(torch.sum(self.running_column_prob,0)) * beta2 + self.running_column_prob*beta) / (torch.sum(self.running_column_prob,0)*(beta2+beta))
        running_row_prob =  (torch.diag(torch.sum(self.running_column_prob,1)) * beta2 + self.running_column_prob*beta) / (torch.sum(self.running_column_prob,1)*(beta2+beta))
        running_relation_logits = self.running_labeling_prob.unsqueeze(0).repeat(self.running_column_prob.shape[0],1)
        running_pred_column_relation_probs = torch.sum(running_relation_logits * running_column_prob,-1)
        running_pred_row_relation_probs = torch.sum(running_relation_logits * running_row_prob,-1)
        running_pred_relation_probs = running_pred_column_relation_probs # +running_pred_row_relation_probs  running_pred_column_relation_probs

        fg_relation_logits = F.softmax(relation_logits, dim=-1).unsqueeze(1).repeat(1, self.running_column_prob.shape[0],1)
        fg_pred_column_relation_probs = torch.sum(fg_relation_logits * running_column_prob,-1)
        fg_pred_row_relation_probs = torch.sum(fg_relation_logits * running_row_prob,-1)
        fg_pred_relation_probs = fg_pred_column_relation_probs#+ fg_pred_row_relation_probs  fg_pred_column_relation_probs


        # fg_relation_logits = F.softmax(fg_pred_column_relation_probs + self.resistance(self.running_label), dim=-1)
        # fg_relation_logits = F.softmax(relation_logits, dim=-1)

        if self.training:
                relation_probs_norm =  fg_pred_relation_probs/ (running_pred_relation_probs + self.eps) 
                #relation_probs_norm =  F.softmax(relation_logits , dim=-1) 
                #relation_probs_norm = F.softmax(relation_logits + self.resistance(self.running_label), dim=-1) 
        else:
                """
                beta = 1
                beta2 = 1
                running_column_prob = (torch.diag(torch.sum(self.running_column_prob,0)) * beta2  + self.running_column_prob*beta) / (torch.sum(self.running_column_prob,0)*( beta2+ beta))
                running_row_prob =  (torch.diag(torch.sum(self.running_column_prob,1))* beta2 + self.running_column_prob*beta) / (torch.sum(self.running_column_prob,1)*( beta2+ beta))
                running_relation_logits = self.running_labeling_prob.unsqueeze(0).repeat(self.running_column_prob.shape[0],1)
                running_pred_column_relation_probs = torch.sum(running_relation_logits * running_column_prob,-1)
                running_pred_row_relation_probs = torch.sum(running_relation_logits * running_row_prob,-1)
                running_pred_relation_probs = running_pred_column_relation_probs  +running_pred_row_relation_probs

                fg_relation_logits = F.softmax(relation_logits, dim=-1).unsqueeze(1).repeat(1, self.running_column_prob.shape[0],1)
                fg_pred_column_relation_probs = torch.sum(fg_relation_logits * running_column_prob,-1)
                fg_pred_row_relation_probs = torch.sum(fg_relation_logits * running_row_prob,-1)
                fg_pred_relation_probs = fg_pred_column_relation_probs + fg_pred_row_relation_probs

                relation_probs_norm = fg_pred_relation_probs/ (running_pred_relation_probs + self.eps) 
                """
                #relation_probs_norm = F.softmax(relation_logits, dim=-1)
                #fg_relation_logits = F.softmax(relation_logits, dim=-1)
                #relation_probs_norm = fg_relation_logits/ (self.running_labeling_prob + self.eps)
                relation_probs_norm =  fg_pred_relation_probs/ (running_pred_relation_probs + self.eps) 

        if self.normalized_probs:
            relation_probs_norm[:, 0] = 1 - relation_probs_norm[:, 1:].sum(1)
            #relation_probs_norm= relation_probs_norm / torch.sum(relation_probs_norm, 1)

        return relation_probs_norm, self.running_labeling_prob.detach(), None if not self.training else torch.sum(rel_labels_one_hot, dim=0)
