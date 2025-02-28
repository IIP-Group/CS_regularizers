import os
import time
import copy
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
import numpy as np
import random

# customized packages
from tools import lplq_regularization_CNN
from tools import quant_CNN
from learning import train_model, plot_training_log
from learning import evaluate_model
from my_models.resnet import resnet20

from tools.quantization_modules import replace_with_quantized, QuantLayer


def main(cfg):
    
    # set random seed for repetition
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    # set global parameters
    operation_on = 'mothim'  # ['local', 'mothim']
    server_name = cfg['server_name'] # ['mothim', 'burmy', 'toros']
    operation_for = 'release'  # ['debug', 'release']
    cuda_parallel = cfg['cuda_parallel']
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['cuda_visible_devices'] #"5"
    
    stage1on = 1
    comment = cfg['comment']
    val_equal_test = cfg['val_equal_test'] # validation set is equal to test set (for tracking metrics)
    exclude_first_last = cfg['exclude_first_last'] 
    
    # pretrained settings
    pretrained = cfg['pretrained']

    batch_size = cfg['batch_size']  # 64
    num_epochs_soft_disc = cfg['num_epochs_soft_disc'] #100 
    num_epochs_disc_retrain = cfg['num_epochs_disc_retrain'] # 40 
    regu_lambda = cfg['regu_lambda']   # 1 is for resnet 34
    num_workers = cfg['num_workers'] # 10 is good for resnet 20 # 4 #16 # 
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(cfg['seed'])
    
    # Optimizer selection (the optimizers of phase-I and phase-II training)
    optimizer1_type = cfg['optimizer1_type']  # ['adam', 'sgd']
    optimizer2_type = cfg['optimizer2_type'] 
    
    lr1, lr2 = cfg['lr1'], cfg['lr2']
    lr1_adjust, lr2_adjust = cfg['lr1_adjust'], cfg['lr2_adjust']
    sgd_momentum, opt_wd = cfg['sgd_momentum'], cfg['opt_wd']
    # don't forget scheduler settings!
    sch1_type = cfg['sch1_type'] # 'cos', 'coswarm', 'step'
    sch2_type = cfg['sch2_type']
    # step size and gamm are for StepLR, eta is for CosineAnnealingLR
    sch1_step_size, sch1_gamma, sch1_eta = cfg['sch1_step_size'], cfg['sch1_gamma'], cfg['sch1_eta']
    sch2_step_size, sch2_gamma, sch2_eta = cfg['sch2_step_size'], cfg['sch2_gamma'], cfg['sch2_eta']
    sch1_T0, sch2_T0 = cfg['sch1_T0'], cfg['sch2_T0']
    sch1_Tmult, sch2_Tmult = cfg['sch1_Tmult'], cfg['sch2_Tmult']

    # phase II settings
    disc_retrain = cfg['disc_retrain']
    
    model_type = cfg['model_type']
    # [resnet18, resnet34, resnet50, vgg16_bn]
    dataset_cat = cfg['dataset_cat']
    # pretrained_flag = True # Todo: modified the paramters of model.resnet18
    
    #% Set data path and transforms
    data_path = f'results/{model_type}-over-{dataset_cat}/'
    os.makedirs(data_path, exist_ok=True)
    # Define training and testing data transforms
    if dataset_cat == 'cifar10':
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        test_transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        all_train_set = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=train_transform)
        all_val_set = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=test_transform)
        test_set = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transform)

    elif dataset_cat == 'imagenet':
        num_classes = 1000 
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if server_name == "mothim": data_dir = "/scratch/share/imagenet"
        else: data_dir = "/scratch2/share2/imagenet" # burmy
        all_train_set = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transform)
        all_val_set = datasets.ImageFolder(root=f'{data_dir}/train', transform=test_transform)
        test_set = datasets.ImageFolder(root=f'{data_dir}/val', transform=test_transform)

    # split train_set into train and validation sets
    val_to_all_ratio = cfg['val_to_all_ratio']
    
    total_samples = len(all_train_set)
    if not val_equal_test:
        train_indices, val_indices = train_test_split(torch.arange(total_samples, dtype=int)[:,None], 
                                                      test_size=val_to_all_ratio)
        train_set = torch.utils.data.Subset(all_train_set, train_indices[:,0])
        val_set = torch.utils.data.Subset(all_val_set, val_indices[:,0])

        train_samples = len(train_set)
        val_samples = len(val_set)
        print(train_samples, val_samples, val_samples/total_samples)
    else: 
        train_samples = len(all_train_set) 
        val_samples = len(test_set)
        train_set = all_train_set
        val_set = test_set
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers, pin_memory=True,
                                                worker_init_fn=seed_worker, generator=g)
    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers, pin_memory=True,
                                                worker_init_fn=seed_worker, generator=g)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers, pin_memory=True,
                                                worker_init_fn=seed_worker, generator=g)
    
    #% Initialize model
    # record start time
    start_time = time.time()
    run_id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    print(f"start time: {run_id}")
    
    num_layers = 0.0  # Initialize
    if model_type == 'resnet18':
        if pretrained:
            model = models.resnet18(weights="IMAGENET1K_V1", num_classes=num_classes)  # weights="IMAGENET1K_V1"
        else:
            model = models.resnet18(num_classes=num_classes)
        num_layers = 18
    elif model_type == 'resnet50':
        if pretrained:
            model = models.resnet50(weights="IMAGENET1K_V2", num_classes=num_classes)  # weights="IMAGENET1K_V2"
        else:
            model = models.resnet50(num_classes=num_classes)
        num_layers = 50
    elif model_type == 'mobilenetv2':
        if pretrained:
            model = models.mobilenet_v2(weights="IMAGENET1K_V2")  # weights="IMAGENET1K_V2"
        else:
            model = models.mobilenet_v2()  # weights="IMAGENET1K_V2"
        num_layers = 53
    elif model_type == 'resnet20':
        model = resnet20()
        num_layers = 20
        if pretrained:
            # model = nn.DataParallel(model, [0]) # to load the DataParallel model
            # model.load_state_dict(torch.load('pretrained_model/resnet20-12fca82f.th')['state_dict'])
            # model = model.module # undo the DataParallel (for now, for consistency)
            # # NEW:
            pretrained_dict = torch.load('pretrained_model/resnet20-12fca82f.th', map_location=torch.device('cpu'))['state_dict']
            nn.modules.utils.consume_prefix_in_state_dict_if_present(
                pretrained_dict, prefix='module.')
            model.load_state_dict(pretrained_dict)
    else:
        print('no other model types.')
    
    #% Set CUDA and reg params
    if cuda_parallel and torch.cuda.device_count() > 1 :
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        device = torch.device("cuda")
        ismodelDP = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # DGX server
        ismodelDP = False
    print("Device:", device)
    model = model.cpu()
    
    # ############ DEBUG BLOCK ################
    # for name, param in model.named_parameters():
    #     print(name, param.size())
    #     print(param.data)
    # ############ DEBUG BLOCK ################
    
    # Regularization parameters
        # 1 is for not-full-normalized-regularization version (not normalized by the number of layers)
    regu_conv = cfg['regu_conv']   # [binary, binary_input_ch, ternary, none]
    regu_fc = cfg['regu_fc']   # [binary, ternary, none]
    schlambda_type = cfg['schlambda_type']
    regu_normal = 'elenum'  # [zero_norm, elenum, none]
    #                         for ternary, select zero_norm;
    #                         for binary, select elenum (though the full-precision result is not as good as zero-norm,
    #                         the quantized result of elenum is better than zero-norm).
    regu_p = 4
    regu_q = 2
    regu_r = 4
    
    # Quantization parameters
    quan_conv_weight = cfg['quan_conv_weight']  # [scaling_binary,scaling_binary_zerotoone,
    #                                        scaling_ternary, scaling_ternary_keepzero]
    quan_fc_weight = cfg['quan_fc_weight']  # [scaling_binary, scaling_binary_zerotoone,
    #                                        scaling_ternary, scaling_ternary_keepzero]
    quan_conv_bias = 'none'
    quan_fc_bias = 'none'
    Quant_conv_weight_by = 'input_ch'   # ['kernel', 'input_ch']
    # Loss function without regularization
    criterion = nn.CrossEntropyLoss().to(device)
    
    #% Start from Pretrained Models
    print("==================================")
    print("Stage 0: Model Pretraining")
    
    pretrained_model = copy.deepcopy(model)
    model = model.to(device)
    test_accuracy_pretrain = evaluate_model(model=model, device=device, loader=test_loader)
    print(f"Test accuracy (pretraining): {test_accuracy_pretrain: .4f}")
    
    #% Stage 1 training and testing
    if optimizer1_type == 'adam':
        optimizer1 = optim.Adam(model.parameters(), lr=lr1, weight_decay=opt_wd)  # Adam optimizer
    elif optimizer1_type == 'sgd':
        optimizer1 = optim.SGD(model.parameters(), lr=lr1, momentum=sgd_momentum, weight_decay=opt_wd)  # SGD optimizer
    elif optimizer1_type == 'radam':
        optimizer1 = optim.RAdam(model.parameters(), lr=lr1, weight_decay=opt_wd)  # Adam optimizer
    
    if sch1_type == 'step': sch1 = StepLR(optimizer1, step_size=sch1_step_size, gamma=sch1_gamma)
    elif sch1_type == 'cos': sch1 = CosineAnnealingLR(optimizer1, T_max=num_epochs_soft_disc, eta_min=sch1_eta)
    elif sch1_type == 'coswarm': sch1 = CosineAnnealingWarmRestarts(optimizer1, T_0=sch1_T0, T_mult=sch1_Tmult,
                                                                                   eta_min=sch1_eta)
    
    print("==================================")
    print('Stage 1: Soft-Discretization Training')
    
    """
        Soft-discretization training
    """
    train_time_soft_disc_start = time.time()
    if stage1on:
        training_log_soft_disc = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,  device=device, best_of="final epoch",
            learning_rate_adjust=lr1_adjust, scheduler=sch1,
            num_epochs=num_epochs_soft_disc,criterion=criterion,optimizer=optimizer1,
            regu_lambda=regu_lambda, regu_p=regu_p, regu_q=regu_q, regu_r=regu_r,
            regu_conv=regu_conv, regu_fc=regu_fc, regu_normal=regu_normal, schlambda_type=schlambda_type,
            exclude_first_last=exclude_first_last, histogram_enabled=False) # complete argument list
    else: 
        training_log_soft_disc = evaluate_model(model=model, loader=test_loader, device=device)
      
    train_accuracy_soft_disc = evaluate_model(model=model, loader=train_loader, device=device)
    test_accuracy_soft_disc = evaluate_model(model=model, loader=test_loader, device=device)
    discretization_loss = lplq_regularization_CNN(
        model=model, op_mode='test', conv_regu=regu_conv, fc_regu=regu_fc,
        normalization_type=regu_normal,
        p=regu_p, q=regu_q, r=regu_r, exclude_first_last=exclude_first_last)
    print(f"Discretization loss:{discretization_loss:.9f}")
    print(f"Train accuracy (soft discretization): {train_accuracy_soft_disc: .4f}")
    print(f"Test accuracy (soft discretization): {test_accuracy_soft_disc: .4f}")
    if stage1on: plot_training_log(training_log=training_log_soft_disc, test_accuracy=test_accuracy_soft_disc) # sue commented out
    
    train_time_soft_disc_end = time.time()
    train_time_soft_disc = train_time_soft_disc_end - train_time_soft_disc_start
    
    # Save the trained model
    trained_model_soft_disc = copy.deepcopy(model)
    
    #% Quantize the parameters and test
    print("================================")
    print('Stage 2: Hard Discretization after Soft-Discretization Training:')
    quant_CNN(input_model=model,
                Quant_conv_weight_by=Quant_conv_weight_by, Quant_conv_weight_mode=quan_conv_weight,
                Quant_fc_weight_by='row', Quant_fc_weight_mode=quan_fc_weight,
                Quant_conv_bias_by=quan_conv_bias, Quant_fc_bias_by=quan_fc_bias,
                exclude_first_last=exclude_first_last)
    
    model = model.to(device)
    
    # quant_kernel_CNN(model)
    test_accuracy_hard_disc = evaluate_model(model=model, loader=test_loader, device=device)
    print(f"Test accuracy (hard discretization): {test_accuracy_hard_disc: .4f}")
    
    # Save the hard-quantized model
    trained_model_hard_disc = copy.deepcopy(model)
    
    #% Stage 3 training
    """
        Discretized network
    """
    original_model = copy.deepcopy(model)
    replace_with_quantized(model, conv_mode=Quant_conv_weight_by, exclude_first_last=exclude_first_last)
    if disc_retrain:
        train_time_disc_retrain_start = time.time()
        print("================================")
        print("Stage 3: Discretized Network Retraining")

        if optimizer2_type == 'adam':
            optimizer2 = optim.Adam(model.parameters(), lr=lr2)
        elif optimizer2_type == 'sgd':
            optimizer2 = optim.SGD(model.parameters(), lr=lr2)
        elif optimizer2_type == 'radam':
            optimizer2 = optim.RAdam(model.parameters(), lr=lr2, weight_decay=opt_wd)  # Adam optimizer
    
        if sch2_type == 'step': sch2 = StepLR(optimizer2, step_size=sch2_step_size, gamma=sch2_gamma)
        elif sch2_type == 'cos': sch2 = CosineAnnealingLR(optimizer2, T_max=num_epochs_disc_retrain, eta_min=sch2_eta)
        elif sch2_type == 'coswarm': sch2 = CosineAnnealingWarmRestarts(optimizer2, T_0=sch2_T0, T_mult=sch2_Tmult, eta_min=sch2_eta)
    
        training_log_disc_retrain = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader, device=device, best_of="accuracy",
            num_epochs=num_epochs_disc_retrain, criterion=criterion, optimizer=optimizer2,
            learning_rate_adjust=lr2_adjust, scheduler=sch2,
            save_path=f"./FDTQ_ckpt/model_disc_retrain.pt", 
            exclude_first_last=exclude_first_last, histogram_enabled=False) # default: no regularization; quantized network has 0 "quantization loss".
        test_accuracy_disc_retrain = evaluate_model(model=model, loader=test_loader, device=device)
        print(f"Test accuracy (discretized network retraining): {test_accuracy_disc_retrain: .4f}")
        plot_training_log(training_log=training_log_disc_retrain, test_accuracy=test_accuracy_disc_retrain) 
        train_time_disc_retrain_end = time.time()
        train_time_disc_retrain = train_time_disc_retrain_end - train_time_disc_retrain_start
    
    else:
        print("No discretized network retraining!")
        training_log_disc_retrain = None
        train_time_disc_retrain = 0
        
    trained_model_disc_retrain = copy.deepcopy(model)
    
    if ismodelDP:
        trained_model_soft_disc = trained_model_soft_disc.module
        trained_model_hard_disc = trained_model_hard_disc.module
        trained_model_disc_retrain = trained_model_disc_retrain.module
    
    #% Training finished, now save
    print("================================")
    print("saving the results")
    save_dict = {
        'device': device,
        'cfg_train': cfg,
        # not in train_config
        'train_time_soft_disc': train_time_soft_disc, 'train_time_disc_retrain': train_time_disc_retrain,
        'trained_model_soft_disc': trained_model_soft_disc.state_dict(),
        'trained_model_hard_disc': trained_model_hard_disc.state_dict(),
        'trained_model_disc_retrain': trained_model_disc_retrain.state_dict(),
        'test_accuracy_soft_disc': test_accuracy_soft_disc,
        'test_accuracy_hard_disc': test_accuracy_hard_disc,
        'test_accuracy_disc_retrain': test_accuracy_disc_retrain,
         ## training log
        'training_log_soft_disc': training_log_soft_disc, 'training_log_disc_retrain': training_log_disc_retrain,
        'train_transform': train_transform, 'test_transform': test_transform,
        'optimizer1': optimizer1.state_dict(), 'optimizer2': optimizer2.state_dict(),
        'sch1': sch1.state_dict(), 'sch2': sch2.state_dict()
    }
    save_to = data_path + f'{comment}-{model_type}-{dataset_cat}-start-{run_id}-{test_accuracy_disc_retrain:.3f}.pth'
    
    torch.save(save_dict, save_to)
    
    print("results are saved to ")
    print(save_to)
    
    print("================================")
    # record end time
    end_time = time.time()
    
    # calculate run time
    run_time = end_time - start_time  # record end time
    
    print(f"Running for {run_time/3600: .2f} hours")
    
    print(f"Soft-Discretization Training: {(train_time_soft_disc_end - train_time_soft_disc_start)/3600: .2f} hrs")
    if disc_retrain:
        print(f"Discretized Network Retraining {(train_time_disc_retrain_end - train_time_disc_retrain_start)/3600: .2f} hrs")
    return save_dict
