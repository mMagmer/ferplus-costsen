"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import sys
sys.path.append('')
#os.chdir("..")

import utils
from data.data_utils import fetch_data, FERDataset, DataLoader, transform_train, transform_infer
from evaluate import evaluate

from efficientnet_pytorch import EfficientNet

#import warnings
#warnings.filterwarnings("ignore", message=".*pthreadpool.*")






def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader), file=sys.stdout) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        # update optimizer parameters
        scheduler.step()
        
        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    """
    Saving & loading of the model.
    """
    parser.add_argument('--experiment_title', default='train model on fer+')
    
    parser.add_argument('--model_dir', default='experiments',
                        help="Directory containing params.json")
    parser.add_argument("--save_name", type=str, default="fer")
    parser.add_argument("--overwrite", action="store_true")
    
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    
    """
    Training Configuration
    """

    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument(
        "--num_train_iter",
        type=int,
        default=1000,
        help="total number of training iterations",
    )
    parser.add_argument(
        "--save_summary_steps", type=int, default=20, help="evaluation frequency"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="total number of batch size of labeled data",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
        help="batch size of evaluation data loader (it does not affect the accuracy)",
    )
    
    """
    Optimizer configurations
    """
    
    parser.add_argument("--opt", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--amp", action="store_true", help="use mixed precision training or not"
    )

    """
    Backbone Net Configurations
    """
    
    parser.add_argument("--net", type=str, default="efficientnet-b0")
    parser.add_argument("--net_from_name", type=bool, default=False)
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--leaky_slope", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)

    """
    Data Configurations
    """
    
    parser.add_argument('--data_dir', default='data/fer2013/fer2013.csv',
                        help="Directory containing the dataset")

    parser.add_argument("--dataset", type=str, default="fer+")
    parser.add_argument("--data_sampler", type=str, default="weigthed")
    parser.add_argument("--num_workers", type=int, default=1)
    
    args = parser.parse_args()

    # Load the parameters from json file
    import json
    
    #print(json.dumps(vars(args),  indent=4))
    
    args.model_dir = os.path.join( args.model_dir,args.save_name)
    print(args.model_dir)
    
    
    
    json_path = os.path.join(args.model_dir, 'params.json')
    
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    assert not (os.path.isfile(json_path) and not args.overwrite), "already existing json configuration file found at {} \
    \n use overwrite flag if you're sure!".format(json_path)
    
    with open(json_path,'w' if args.overwrite else 'x' ) as f:
        json.dump(vars(args), f, indent=4)
    
    
    params = utils.Params(json_path)
    
    
    # use GPU if available
    params.cuda = torch.cuda.is_available()
    

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    
    # hyper parameter settings
    logging.info("Setup for training model:")
    logging.info((json.dumps(vars(args),  indent=4)))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # fetch dataloaders
    data_splits ,classes = fetch_data()
    trainset = FERDataset(data_splits['train'],transform=transform_train)
    valset = FERDataset(data_splits['val'],transform=transform_infer)
    
    blance_sampler = None
    
    train_dl = DataLoader(trainset, batch_size= params.batch_size, 
                          shuffle=True, sampler= blance_sampler,
                          num_workers= params.num_workers, pin_memory= params.cuda)
    
    val_dl = DataLoader(valset, batch_size= params.batch_size, 
                        shuffle= False, sampler= None,
                        num_workers= 2, pin_memory= params.cuda)

    logging.info("- done.")

    # Define the model and optimizer
            
    if "efficientnet" in params.net:
        if params.pretrained:
            raise Exception("Not Implemented Yet!")

        else:
            logging.info("Using not pretrained model "+ params.net+ " ...")
            model = EfficientNet.from_name('efficientnet-b0',in_channels=trainset.in_channels,num_classes=trainset.num_classes)

    else:
        raise Exception("Not Implemented Error! check --net ")
       
    
    model = model.cuda() if params.cuda else model
    optimizer = optim.SGD(model.parameters(),
                          lr=params.lr,momentum=params.momentum,weight_decay=params.weight_decay, nesterov=True)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.6)

    #assert False, 'forced stop!'
    # fetch loss function and metrics
    loss_fn = torch.nn.CrossEntropyLoss()
    
    def accuracy(outputs, labels):
        """
        Compute the accuracy, given the outputs and labels for all images.
        Args:
            outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
            labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
        Returns: (float) accuracy in [0,1]
        """
        outputs = np.argmax(outputs, axis=1)
        return np.sum(outputs==labels)/float(labels.size)


    # maintain all metrics required in this dictionary- these are used in the training and evaluation loops
    metrics = {
        'accuracy': accuracy,
        # could add more metrics such as accuracy for each token type
    }
    # metrics.accuracy_score, sklearn.metrics.recall_score, sklearn.metrics.f1_score, sklearn.metrics.precision_score,
    #or sklearn.metrics.classification_report  for all

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, scheduler, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)