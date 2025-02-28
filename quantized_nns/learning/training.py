import os

from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import copy

from learning.evaluation import evaluate_model
from tools.regularization_modules import lplq_regularization_CNN

def train_model(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader, 
                num_epochs:int=10, criterion=None, optimizer=None,
                learning_rate_adjust: bool=True, scheduler=None,
                regu_lambda:float=0., regu_p: int=6, regu_q: int=2, regu_r: int=2,
                regu_conv:str="", regu_fc: str="",
                regu_normal: str="", 
                exclude_first_last=True,
                best_of:str="loss", device:str="cpu", 
                save_path:str=f"./ckpt/model.pt",
                histogram_enabled:bool=False,
                schlambda_type=None, lambda_min=0):
    """
    Trains the model on the given dataset. Selects the best model based on the
    validation set and saves it to the given path. 
    Inputs: 
        model: The model to train [nn.Module]
        train_loader: The training data loader [DataLoader]
        val_loader: The validation data loader [DataLoader]
        num_epochs: The number of epochs to train for [int]
        criterion: The loss function [Any]
        optimizer: The optimizer [Any]
        best_of: The metric to use for validation [str: "loss" or "accuracy" or "final epoch"]
        device: The device to train on [str: cpu, cuda, or mps]
        save_path: The path to save the model to [str]
    Output:
        Dictionary containing the training and validation losses and accuracies
        at each epoch. Also contains the epoch number of the best model.
    """

    # Check that the best_of parameter is valid
    best_of = best_of.lower()
    assert best_of in ["loss", "accuracy", "final epoch"], "best_of must be 'loss', 'accuracy', or 'final epoch'"
    print("Validation metric:", best_of)

    # Set the best validation metric to \pm infinity
    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    best_val_epoch = 0
    best_dict = model.state_dict()

    # Logs for the training
    log_train_acc, log_train_loss = list(), list()
    log_val_acc, log_val_loss = list(), list()

    log_train_reg, log_train_crit = [], [] # SUE

    # Set the device for the model
    model.to(device)
    print(f"Training on device: {device}")

    pbar = tqdm(range(num_epochs), desc="Epochs")

    num_batches = len(train_loader)

    lambda_max = regu_lambda
    if schlambda_type == 'cos':
        regu_lambda_list = lambda_min  + 1/2*(lambda_max - lambda_min)*(1+np.cos(np.arange(num_epochs)/num_epochs*np.pi))
        regu_lambda_list = regu_lambda_list[::-1]
    if schlambda_type == 'exp': 
        log_lambda_max = np.log10(lambda_max)
    
    for epoch in pbar:

        ###############################
        # Train the model for one epoch
        ###############################
        if schlambda_type is not None:
            if schlambda_type == 'cos':
                regu_lambda = regu_lambda_list[epoch]
            elif schlambda_type == 'lin':
                regu_lambda = lambda_min + (lambda_max - lambda_min)*(epoch+1)/num_epochs
                # print(f'New lambda: {regu_lambda :.4f}')
            elif schlambda_type == 'exp':
                regu_lambda = 10**(log_lambda_max*(epoch+1)/num_epochs)
                # print(f'New lambda: {regu_lambda :.4f}')
            else: raise RuntimeWarning('Undefined sch lambda')
    
        # Keep track of the number of correct predictions and loss
        all_corrects, all_samples = 0, 0
        total_loss = 0.0
        total_criterion, total_reg = 0.0,0.0

        # Set the model to training mode
        model.train()
        
        for batch_idx, (images, labels) in enumerate(train_loader):

            # Move the images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get the predictions
            _, preds = torch.max(outputs, dim=1)

            # Store the number of correct predictions
            corrects = torch.sum(preds == labels.data)

            # Compute the loss
            loss = criterion(outputs, labels)
            # print('loss criterion', loss)
            total_criterion += loss.item()*labels.shape[0]
            if regu_lambda != 0:
                reg = regu_lambda * lplq_regularization_CNN(model=model, op_mode='train',
                                                                                 conv_regu=regu_conv, fc_regu=regu_fc,
                                                                                 normalization_type=regu_normal,
                                                                                 p=regu_p, q=regu_q, r=regu_r,
                                                                                 exclude_first_last=exclude_first_last)
                loss += reg
                total_reg += reg.item()
                # print('loss reg', reg)
            # print('total loss: ', loss,' at', batch_idx, '\n')
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update loss
            total_loss += loss.item()*len(labels)

            # Update the weights
            optimizer.step()

            # Update the number of correct predictions and samples
            all_corrects += corrects
            all_samples += len(labels)
            if learning_rate_adjust and isinstance(scheduler, CosineAnnealingWarmRestarts):       
                scheduler.step(epoch + batch_idx/num_batches)

        # Compute the training accuracy for the epoch
        train_acc = float(all_corrects / all_samples)

        # Compute the training loss for the epoch
        train_loss = total_loss / all_samples

        # SUE: Compute the total for criterion loss and regularizer
        train_reg = total_reg / len(train_loader)
        train_criterion = total_criterion / all_samples

        # Log the training accuracy and loss
        log_train_acc.append(train_acc)
        log_train_loss.append(train_loss)

        log_train_reg.append(train_reg)
        log_train_crit.append(train_criterion)

        ##################################
        # Validate the model for one epoch
        ##################################
        
        # Keep track of the number of correct predictions and loss
        all_corrects, all_samples = 0, 0
        total_loss = 0.0

        # Set the model to evaluation mode
        model.eval()

        with torch.no_grad():
        
            for images, labels in val_loader:

                # Move the images and labels to the device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Get the predictions
                _, preds = torch.max(outputs, dim=1)

                # Compute the loss
                if regu_lambda == 0.:
                    loss=criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)+regu_lambda*lplq_regularization_CNN(
                    model=model, op_mode="test", conv_regu=regu_conv, fc_regu=regu_fc,
                    normalization_type=regu_normal,
                    p=regu_p, q=regu_q, r=regu_r, 
                    exclude_first_last=exclude_first_last)

                # Update loss
                total_loss += loss.item()*len(labels)

                # Store the number of correct predictions
                corrects = torch.sum(preds == labels.data)

                # Update the number of correct predictions and samples
                all_corrects += corrects
                all_samples += len(labels)

        # Compute the training accuracy for the epoch
        val_acc = float(all_corrects / all_samples)

        # Compute the training loss for the epoch
        val_loss = total_loss / all_samples

        # Log the training accuracy and loss
        log_val_acc.append(val_acc)
        log_val_loss.append(val_loss)

        # Update the best model

        if best_of == "loss":
            # Save the model if the validation loss has decreased
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                best_dict = model.state_dict()
        elif best_of == "accuracy":
            # Save the model if the validation accuracy has increased
            if val_acc > best_val_acc:
                # best_val_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                best_dict = copy.deepcopy(model.state_dict())
        else: # final epoch
            if epoch == num_epochs:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        if val_acc > best_val_acc: # only for printing
            best_val_acc = val_acc
            best_val_epoch = epoch
      
        if learning_rate_adjust and isinstance(scheduler, (StepLR, CosineAnnealingLR)):       
            scheduler.step()
        # Update the progress bar
        pbar.set_description(f"TR L|A: {train_loss:.4f}|{train_acc:.4f}," 
                           + f" VL L|A: {val_loss:.4f}|{val_acc:.4f}"
                           + f" Best Epoch: {best_val_epoch} w{best_val_acc:.4f}")

    training_log = {
        "train_acc": log_train_acc,
        "train_loss": log_train_loss,
        "val_acc": log_val_acc,
        "val_loss": log_val_loss,
        "best_val_epoch": best_val_epoch,
        'train_reg': log_train_reg, 
        'train_crit': log_train_crit
    }

    if best_of == 'accuracy' or best_of == 'loss':
        model.load_state_dict(best_dict) # the best_dict gets updated with the model, so this does not do what i want

    return training_log




def plot_training_log(training_log, test_accuracy, show_baseline=True):
    epochs = range(1, len(training_log['train_acc']) + 1)

    plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_log['train_acc'], label='Train Acc')
    plt.plot(epochs, training_log['val_acc'], label='Val Acc')
    if show_baseline:
        plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plt.subplot(1, 2, 2)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, training_log['train_loss'], label='Train Loss')
    plt.plot(epochs, training_log['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, training_log['train_crit'], label='Train Crit')
    plt.title('Training Crit')
    plt.xlabel('Epoch')
    plt.ylabel('Crit')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(epochs, np.log10(np.array(training_log['train_reg'])), label='Train Reg')
    plt.title('Training Reg')
    plt.xlabel('Epoch')
    plt.ylabel('Reg')
    plt.legend()

    plt.tight_layout()
    plt.show(block=False)