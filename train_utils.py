#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional
from torch.nn.modules.loss import _Loss


class MarginCalibratedCELoss(_Loss):
    r"""This criterion computes the cross entropy loss between input and target.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.
    
    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        margin (Tensor, optional): a manual margin calibration applied peredicted output.
            If given, has to be a Tensor of size `C`
        smoothing_dist (Tensor, optional): dist that use for max entropy regularizer.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0].
    """
    
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float
        
    def __init__(self, weight: Optional[Tensor] = None, margin: Optional[Tensor] = None,
                 smoothing_dist: Optional[Tensor] = None, ignore_index: int = -100,
                 size_average=None, reduce=None, reduction: str = 'sum', label_smoothing: float = 0.0) -> None:
        super(MarginCalibratedCELoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('margin', margin)
        self.register_buffer('smoothing_dist', smoothing_dist)
        self.weight: Optional[Tensor]
        self.margin: Optional[Tensor]
        self.smoothing_dist: Optional[Tensor]
            
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.it = 50
        self.T = 50
        #self.step()
    
    def step(self):
        if self.it < self.T:
            #self.margin = self.margin*self.it/50
            #self.weight = self.weight**(self.it/50)
            self.it+=1
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if not self.margin is None:
            input = input + self.margin*self.it/self.T
        
        N = 1
        #if self.reduction=='mean':
        N = target.shape[0]
        #self.reduction='sum'
            
        loss = F.cross_entropy(input, target, weight=self.weight**(self.it/self.T),
                               ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=0.0)
        
        max_entropy_regularizer = 0
        if not self.smoothing_dist is None:
            C = len(self.smoothing_dist)
            max_entropy_regularizer = F.cross_entropy(input, 0*F.one_hot(target, num_classes=C)+self.smoothing_dist,
                                                      weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        
        return ( (1-self.label_smoothing)*loss + self.label_smoothing*max_entropy_regularizer)/N

    
import numpy as np
from typing import Dict

class ConfusionMatrix():
    """Calculates confusion matrix for multi-class data.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must contain logits and has the following shape (batch_size, num_classes, ...).
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices.
       During the computation, argmax of `y_pred` is taken to determine predicted classes.
       
    Args:
        num_classes: Number of classes, should be > 1. See notes for more details.
    """

    def __init__(self,num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes),dtype = 'int')
        self._num_examples = 0
    
    def reset(self) -> None:
        self.confusion_matrix.fill(0)
        self._num_examples = 0
            
    def update(self, outputs, labels) -> None:
        y_pred, y = np.argmax(outputs, axis=1).flatten(), labels

        self._num_examples += y_pred.shape[0]

        # target is (batch_size, ...)
        y = y.flatten()

        target_mask = (y >= 0) & (y < self.num_classes)
        y = y[target_mask]
        y_pred = y_pred[target_mask]

        cm = np.zeros_like(self.confusion_matrix)
        for i , j in zip(y,y_pred):
            cm[i,j] += 1
            
        self.confusion_matrix += cm
        
    def compute(self, average: bool = True) -> Dict:
        cm = self.confusion_matrix
        accuracy = cm.diagonal().sum() / (cm.sum() + 1e-15)
        recall = cm.diagonal() / (cm.sum(axis=1) + 1e-15)
        precision = cm.diagonal() / (cm.sum(axis=0) + 1e-15)  
        iou = cm.diagonal() / (cm.sum(axis=1) + cm.sum(axis=0) - cm.diagonal() + 1e-15)
        
        if average:
            precision = precision.mean()
            recall = recall.mean()
            iou = iou.mean()
        
        metrics = {'accuracy': accuracy,
                   'recall' : recall,
                   'precision' : precision,
                   'IoU' : iou}
        
        return metrics    
    
    
if __name__ == "__main__":
    
    p = torch.Tensor([36.3419, 26.4458, 12.5597, 12.4088,  8.6819,  0.6808,  2.2951,  0.5860])
    gmean = lambda p: torch.exp(torch.log(p).mean())
    w = (p/gmean(p))**-1
    weight = w**(1/3)
    margin = -torch.log(w**(1/3))

    criterion = MarginCalibratedCELoss(weight=weight,margin=margin)
    t = torch.randint(8,(20,))
    i = torch.rand((20,8), requires_grad=True)
    loss = criterion(i, t)
    loss.backward()