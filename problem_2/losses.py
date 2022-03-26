import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import pickle
import os

from util import sample_values
from util import make_sampling_array

import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive

class NCELoss(nn.Module):
    """ Class for calculating of the noise-contrasting estimation loss. """

    def __init__(self, opt, vocab_size):
        super(NCELoss, self).__init__()
        # Initialize parameters
        self.vocab_size = vocab_size
        self.opt = opt

        # Initialize the sampling array and the class probability dictionary
        if os.path.isfile(self.opt.array_path):
            print('Loading sampling array from the pickle %s.' %
                  self.opt.array_path)
            with open(self.opt.array_path, 'rb') as f:
                self.sampling_array, self.class_probs = pickle.load(f)
        else:
            self.sampling_array, self.class_probs = make_sampling_array(
                self.vocab_size, self.opt.array_path)

    def forward(self, inputs, labels, weights, biases, sampled_values=None):
        """ Performs the forward pass. If sampled_values is None, a log uniform candidate sampler is used
        to obtain the required values. """

        # SHAPES:
        # inputs shape=[batch_size, dims]
        # flat_labels has shape=[batch_size * num_true]
        # sampled_candidates has shape=[num_sampled]
        # true_expected_count has shape=[batch_size, num_true]
        # sampled_expected_count has shape=[num_sampled]
        # all_ids has shape=[batch_size * num_true + num_sampled]
        # true_w has shape=[batch_size * num_true, dims]
        # true_b has shape=[batch_size * num_true]
        # sampled_w has shape=[num_sampled, dims]
        # sampled_b has shape=[num_sampled]
        # row_wise_dots has shape=[batch_size, num_true, dims]
        # dots_as_matrix as size=[batch_size * num_true, dims]
        # true_logits has shape=[batch_size, num_true]
        # sampled_logits has shape=[batch_size, num_sampled]

        flat_labels = labels.view([-1])
        num_true = labels.size()[1]
        true_per_batch = flat_labels.size()[0]
        print('Obtaining sampled values ...')
        if sampled_values is None:
            # Indices representing the data classes have to be sorted in the order of descending frequency
            # for the sampler to provide representative distractors and frequency counts
            sampled_values = sample_values(labels, self.opt.num_sampled,
                                           self.opt.unique,
                                           self.opt.remove_accidental_hits,
                                           self.sampling_array,
                                           self.class_probs)
        # Stop gradients for the sampled values
        sampled_candidates, true_expected_count, sampled_expected_count = (
            s.detach() for s in sampled_values)

        print('Calculating the NCE loss ...')
        # Concatenate true and sampled labels
        all_ids = torch.cat((flat_labels, sampled_candidates), 0)
        # Look up the embeddings of the combined labels
        all_w = torch.index_select(weights, 0, all_ids)
        all_b = torch.index_select(biases, 0, all_ids)
        # Extract true values
        true_w = all_w[:true_per_batch, :]
        true_b = all_b[:true_per_batch]
        # Extract sampled values
        sampled_w = all_w[true_per_batch:, :]
        sampled_b = all_b[true_per_batch:]
        # Obtain true logits
        tw_c = true_w.size()[1]
        true_w = true_w.view(-1, num_true, tw_c)
        row_wise_dots = inputs.unsqueeze(1) * true_w
        dots_as_matrix = row_wise_dots.view(-1, tw_c)
        true_logits = torch.sum(dots_as_matrix, 1).view(-1, num_true)
        true_b = true_b.view(-1, num_true)
        true_logits += true_b.expand_as(true_logits)
        # Obtain sampled logits; @ is the matmul operator
        sampled_logits = inputs @ sampled_w.t()
        sampled_logits += sampled_b.expand_as(sampled_logits)

        if self.opt.subtract_log_q:
            print('Subtracting log(Q(y|x)) ...')
            # Subtract the log expected count of the labels in the sample to get the logits of the true labels
            true_logits -= torch.log(true_expected_count)
            sampled_logits -= torch.log(
                sampled_expected_count.expand_as(sampled_logits))

        # Construct output logits and labels
        out_logits = torch.cat((true_logits, sampled_logits), 1)
        # Divide true logit labels by num_true to ensure the per-example labels sum to 1.0,
        # i.e. form a proper probability distribution.
        true_logit_labels = torch.ones(true_logits.size()) / num_true
        sampled_logit_labels = torch.zeros(sampled_logits.size())
        out_labels = torch.cat((true_logit_labels, sampled_logit_labels), 1)
        out_labels = Variable(out_labels)

        # Calculate the sampled losses (equivalent to TFs 'sigmoid_cross_entropy_with_logits')
        loss_criterion = nn.BCELoss()
        nce_loss = loss_criterion(torch.sigmoid(out_logits), out_labels)
        return nce_loss

# import torch
# import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive

# import torch
# import torch.nn.functional as F
# from torch import nn

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


# import math

# import torch
# import torch.nn as nn
# from .alias_multinomial import AliasMultinomial

# # A backoff probability to stabilize log operation
# BACKOFF_PROB = 1e-10

# class NCELoss(nn.Module):
#     """Noise Contrastive Estimation
#     NCE is to eliminate the computational cost of softmax
#     normalization.
#     There are 3 loss modes in this NCELoss module:
#         - nce: enable the NCE approximation
#         - sampled: enabled sampled softmax approximation
#         - full: use the original cross entropy as default loss
#     They can be switched by directly setting `nce.loss_type = 'nce'`.
#     Ref:
#         X.Chen etal Recurrent neural network language
#         model training with noise contrastive estimation
#         for speech recognition
#         https://core.ac.uk/download/pdf/42338485.pdf
#     Attributes:
#         noise: the distribution of noise
#         noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
#         norm_term: the normalization term (lnZ in paper), can be heuristically
#         determined by the number of classes, plz refer to the code.
#         reduction: reduce methods, same with pytorch's loss framework, 'none',
#         'elementwise_mean' and 'sum' are supported.
#         loss_type: loss type of this module, currently 'full', 'sampled', 'nce'
#         are supported
#     Shape:
#         - noise: :math:`(V)` where `V = vocabulary size`
#         - target: :math:`(B, N)`
#         - loss: a scalar loss by default, :math:`(B, N)` if `reduction='none'`
#     Input:
#         target: the supervised training label.
#         args&kwargs: extra arguments passed to underlying index module
#     Return:
#         loss: if `reduction='sum' or 'elementwise_mean'` the scalar NCELoss ready for backward,
#         else the loss matrix for every individual targets.
#     """

#     def __init__(self,
#                  noise,
#                  noise_ratio=100,
#                  norm_term='auto',
#                  reduction='elementwise_mean',
#                  per_word=False,
#                  loss_type='nce',
#                  ):
#         super(NCELoss, self).__init__()

#         # Re-norm the given noise frequency list and compensate words with
#         # extremely low prob for numeric stability
#         probs = noise / noise.sum()
#         probs = probs.clamp(min=BACKOFF_PROB)
#         renormed_probs = probs / probs.sum()

#         self.register_buffer('logprob_noise', renormed_probs.log())
#         self.alias = AliasMultinomial(renormed_probs)

#         self.noise_ratio = noise_ratio
#         if norm_term == 'auto':
#             self.norm_term = math.log(noise.numel())
#         else:
#             self.norm_term = norm_term
#         self.reduction = reduction
#         self.per_word = per_word
#         self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
#         self.ce = nn.CrossEntropyLoss(reduction='none')
#         self.loss_type = loss_type

#     def forward(self, target, *args, **kwargs):
#         """compute the loss with output and the desired target
#         The `forward` is the same among all NCELoss submodules, it
#         takes care of generating noises and calculating the loss
#         given target and noise scores.
#         """

#         batch = target.size(0)
#         max_len = target.size(1)
#         if self.loss_type != 'full':

#             noise_samples = self.get_noise(batch, max_len)

#             # B,N,Nr
#             logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
#             logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

#             # (B,N), (B,N,Nr)
#             logit_target_in_model, logit_noise_in_model = self._get_logit(target, noise_samples, *args, **kwargs)

#             if self.loss_type == 'nce':
#                 if self.training:
#                     loss = self.nce_loss(
#                         logit_target_in_model, logit_noise_in_model,
#                         logit_noise_in_noise, logit_target_in_noise,
#                     )
#                 else:
#                     # directly output the approximated posterior
#                     loss = - logit_target_in_model
#             elif self.loss_type == 'sampled':
#                 loss = self.sampled_softmax_loss(
#                     logit_target_in_model, logit_noise_in_model,
#                     logit_noise_in_noise, logit_target_in_noise,
#                 )
#             # NOTE: The mix mode is still under investigation
#             elif self.loss_type == 'mix' and self.training:
#                 loss = 0.5 * self.nce_loss(
#                     logit_target_in_model, logit_noise_in_model,
#                     logit_noise_in_noise, logit_target_in_noise,
#                 )
#                 loss += 0.5 * self.sampled_softmax_loss(
#                     logit_target_in_model, logit_noise_in_model,
#                     logit_noise_in_noise, logit_target_in_noise,
#                 )

#             else:
#                 current_stage = 'training' if self.training else 'inference'
#                 raise NotImplementedError(
#                     'loss type {} not implemented at {}'.format(
#                         self.loss_type, current_stage
#                     )
#                 )

#         else:
#             # Fallback into conventional cross entropy
#             loss = self.ce_loss(target, *args, **kwargs)

#         if self.reduction == 'elementwise_mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

#     def get_noise(self, batch_size, max_len):
#         """Generate noise samples from noise distribution"""

#         noise_size = (batch_size, max_len, self.noise_ratio)
#         if self.per_word:
#             noise_samples = self.alias.draw(*noise_size)
#         else:
#             noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(*noise_size)

#         noise_samples = noise_samples.contiguous()
#         return noise_samples

#     def _get_logit(self, target_idx, noise_idx, *args, **kwargs):
#         """Get the logits of NCE estimated probability for target and noise
#         Both NCE and sampled softmax Loss are unchanged when the probabilities are scaled
#         evenly, here we subtract the maximum value as in softmax, for numeric stability.
#         Shape:
#             - Target_idx: :math:`(N)`
#             - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
#         """

#         target_logit, noise_logit = self.get_score(target_idx, noise_idx, *args, **kwargs)

#         target_logit = target_logit.sub(self.norm_term)
#         noise_logit = noise_logit.sub(self.norm_term)
#         return target_logit, noise_logit

#     def get_score(self, target_idx, noise_idx, *args, **kwargs):
#         """Get the target and noise score
#         Usually logits are used as score.
#         This method should be override by inherit classes
#         Returns:
#             - target_score: real valued score for each target index
#             - noise_score: real valued score for each noise index
#         """
#         raise NotImplementedError()

#     def ce_loss(self, target_idx, *args, **kwargs):
#         """Get the conventional CrossEntropyLoss
#         The returned loss should be of the same size of `target`
#         Args:
#             - target_idx: batched target index
#             - args, kwargs: any arbitrary input if needed by sub-class
#         Returns:
#             - loss: the estimated loss for each target
#         """
#         raise NotImplementedError()

#     def nce_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
#         """Compute the classification loss given all four probabilities
#         Args:
#             - logit_target_in_model: logit of target words given by the model (RNN)
#             - logit_noise_in_model: logit of noise words given by the model
#             - logit_noise_in_noise: logit of noise words given by the noise distribution
#             - logit_target_in_noise: logit of target words given by the noise distribution
#         Returns:
#             - loss: a mis-classification loss for every single case
#         """

#         # NOTE: prob <= 1 is not guaranteed
#         logit_model = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
#         logit_noise = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

#         # predicted probability of the word comes from true data distribution
#         # The posterior can be computed as following
#         # p_true = logit_model.exp() / (logit_model.exp() + self.noise_ratio * logit_noise.exp())
#         # For numeric stability we compute the logits of true label and
#         # directly use bce_with_logits.
#         # Ref https://pytorch.org/docs/stable/nn.html?highlight=bce#torch.nn.BCEWithLogitsLoss
#         logit_true = logit_model - logit_noise - math.log(self.noise_ratio)

#         label = torch.zeros_like(logit_model)
#         label[:, :, 0] = 1

#         loss = self.bce_with_logits(logit_true, label).sum(dim=2)
#         return loss

#     def sampled_softmax_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
#         """Compute the sampled softmax loss based on the tensorflow's impl"""
#         logits = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
#         q_logits = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)
#         # subtract Q for correction of biased sampling
#         logits = logits - q_logits
#         labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()
#         loss = self.ce(
#             logits.view(-1, logits.size(-1)),
#             labels.view(-1),
#         ).view_as(labels)

#         return loss

# class ContrastiveLoss(T.nn.Module):
#   def __init__(self, m=2.0):
#     super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
#     self.m = m  # margin or radius

#   def forward(self, y1, y2, d=0):
#     # d = 0 means y1 and y2 are supposed to be same
#     # d = 1 means y1 and y2 are supposed to be different

#     euc_dist = T.nn.functional.pairwise_distance(y1, y2)

#     if d == 0:
#       return T.mean(T.pow(euc_dist, 2))  # distance squared
#     else:  # d == 1
#       delta = self.m - euc_dist  # sort of reverse distance
#       delta = T.clamp(delta, min=0.0, max=None)
#       return T.mean(T.pow(delta, 2))  # mean over all rows


# class NCELoss():
#     pass
