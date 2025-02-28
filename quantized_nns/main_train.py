import yaml

# customized packages
from train_from_cfg import main

#%
# Choose network model: 'resnet20' for CIFAR10, 'resnet18' for ImageNet
network_model = 'resnet20' # 'resnet20' or 'resnet18'
# Choose quantization mode: 'binary' or 'ternary'
quant_mode = 'binary'

with open(f'cfg_train_{network_model}.yaml', 'rt') as f:
    cfg = yaml.safe_load(f.read())

# Set these according to the device you are training on
cfg['server_name'] = 'burmy'# ['mothim', 'burmy', 'toros']
cfg['cuda_visible_devices'] = '0'  #"5"

# The name of the saved file will start with your 'comment'
cfg['comment'] = 'binary'
# Set the lambda: 10 for binary, 1e5 for ternary
cfg['regu_lambda'] = 10 # 10, 1e5
# Set the number of epochs for each step of training: 40, 20 for CIFAR-10; 400, 20 for ImageNet
cfg['num_epochs_soft_disc'], cfg['num_epochs_disc_retrain'] = 40, 20

# Keep lambda constant
cfg['schlambda_type'] = None # no lambda scheduling

if quant_mode == 'ternary': # otherwise use the default cfg file assumes binary
    cfg['regu_conv'] = 'ternary_input_ch'
    cfg['regu_fc'] = 'ternary'
    cfg['quan_conv_weight'] = 'scaling_ternary'
    cfg['quan_fc_weight'] = 'scaling_ternary'

# Start training
out = main(cfg)

# Test accuracy with the final quantized network
print('Test accuracy: ', out['test_accuracy_disc_retrain'])
# You can extract other information of the model after each training stage from out

    
