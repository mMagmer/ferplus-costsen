"""Evaluates the model"""

import argparse
import logging
import os
from tabulate import tabulate

import numpy as np
import torch
from torch.autograd import Variable
import utils

from data.data_utils import fetch_data , FERDataset , transform_train , transform_infer

from train_utils import MarginCalibratedCELoss , ConfusionMatrix

@torch.no_grad()
def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: main metric for determining best performing model
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    cm = ConfusionMatrix(num_classes=8)
    
    summ = []
    loss_avg = utils.RunningAverage()

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.cpu().detach().numpy()
        labels_batch = labels_batch.cpu().detach().numpy()

        # compute all metrics on this batch
        cm.update(output_batch,labels_batch)
        
        #summary_batch = {metric: metrics[metric](output_batch, labels_batch)
        #                 for metric in metrics}
        #summary_batch['loss'] = loss.item()
        #summ.append(summary_batch)
        
        loss_avg.update(loss.item())

    # compute mean of all metrics in summary
    metrics_mean = cm.compute()
    metrics_mean['loss'] = loss_avg()
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    
    # compute per class metrics in summary
    metrics_per_class = cm.compute(average=False)
    metric_list = ['recall', 'precision', 'IoU']
    logging.info("-per class Eval metrics :")
    classes = dataloader.dataset.classes
    logging.info(tabulate([[m , *[b.item() for b in metrics_per_class[m]]] for m in metric_list],
                          headers=['Metric',*classes], tablefmt="rst",floatfmt=".3f"))
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
