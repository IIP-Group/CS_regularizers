import torch
import torch.nn as nn


def discretize_to_ternary(x: torch.Tensor):
    # x MUST CONTAIN unique values which must be true for the weights 
    # x: tensor consisting of values centered around 0, +- alpha
    # If x_i <= threshold, x_i \to 0; otherwise, x_i \to alpha
    # For a given threshold, alpha = l1norm(entries above threshold) / length(entries above threshold)
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs.flatten(), descending=True)
    N = x_sorted.shape[0]
    threshold_candidates = x_sorted
    x_cumsum = torch.cumsum(x_sorted, dim=0)
    alpha_candidates = x_cumsum / torch.arange(1, N+1)
    objective_candidates = x_cumsum**2 / torch.arange(1, N+1)
    
    idx_star = torch.argmax(objective_candidates)
    threshold = threshold_candidates[idx_star]
    alpha = alpha_candidates[idx_star]

    disced_x = torch.zeros_like(x)
    setI = x_abs >= threshold
    disced_x[setI] = alpha * torch.sign(x[setI])
    signs = torch.sign(disced_x)
    return disced_x, alpha, signs, threshold

#%% Test block
if __name__ == '__main__':
    from matplotlib import pyplot as plt 
    x = torch.tensor([-2, -2, -0.1,  0,0.1,1,  2, 2.0])
    print(discretize_to_ternary(x))
    n = 10000
    mu = 10
    x = torch.cat((torch.randn((n)) - mu, torch.randn((n//5)), 
                torch.randn((n)) + mu))
    # x = torch.tensor([-2.0,-2.0,  2.0, 2.0])*1e-2 #+ torch.randn(6)*1e-5
    plt.figure()
    plt.hist(x.data, 100)
    
    disced_x, alpha, signs, threshold = discretize_to_ternary(x)
    plt.figure()
    plt.hist(disced_x.data, 100)

#%%
# Define the quantization operation (after training) for CNN
def quant_CNN(input_model, Quant_conv_weight_by='kernel', Quant_conv_weight_mode='scaling_ternary',
              Quant_fc_weight_by='row', Quant_fc_weight_mode='scaling_ternary',
              Quant_conv_bias_by='none', Quant_fc_bias_by='none',
              exclude_first_last=True):
    input_model = input_model.cpu()
    for name, child in input_model.named_modules():
        # if name != 'fc.weight' and name != 'conv1.weight':
        if not isinstance(child, (nn.Conv2d, nn.Linear)): 
            continue
        if exclude_first_last and ((isinstance(child, nn.Conv2d) and child.in_channels == 3)
                                or (isinstance(child, nn.Linear) and (child.out_features == 10 or child.out_features == 1000))):
            continue
        if isinstance(child, nn.Linear):
            find_sfactors_qweights(child, quant_mode=Quant_fc_weight_mode, conv_mode=Quant_fc_weight_by,
                                    quantize_inplace=True)
        else:
            find_sfactors_qweights(child, quant_mode=Quant_conv_weight_mode, conv_mode=Quant_conv_weight_by,
                                    quantize_inplace=True)


def find_sfactors_qweights(layer, quant_mode='binary', conv_mode='input_ch', quantize_inplace=False):
    """"
        Find scaling factors and quantized weights for binary and ternary
    """
    assert isinstance(layer, (nn.Conv2d, nn.Linear))
    fp_weight = layer.weight.data
    with torch.no_grad():
        if 'binary' in quant_mode: # it might be called scaling_binary
            if isinstance(layer, nn.Linear):  dim = 1
            else: # conv2d
                dim = (1,2,3) if conv_mode == 'input_ch' else (2,3)
            scaling_factors = torch.mean(torch.abs(fp_weight), dim=dim, keepdim=True)
            q_weights = torch.sign(fp_weight) 

        elif 'ternary' in quant_mode: # it might be called scaling _ternary
            if isinstance(layer, nn.Linear) or conv_mode == 'input_ch':
                if isinstance(layer, nn.Linear):  dim = (layer.in_features, 1)
                else: # conv2d
                    dim = (layer.out_channels, 1, 1, 1)
                scaling_factors = torch.zeros(dim, device=layer.weight.device)
                q_weights = torch.zeros_like(fp_weight)
                for i in range(fp_weight.shape[0]):
                    disced, scaling_factors[i], q_weights[i], tau = discretize_to_ternary(fp_weight[i])

            elif conv_mode == 'kernel':
                dim = (layer.out_channels, layer.in_channels, 1, 1)
                scaling_factors = torch.zeros(dim, device=fp_weight.device)
                q_weights = torch.zeros_like(fp_weight)
                for i in range(fp_weight.shape[0]):
                    for j in range(fp_weight.shape[1]):
                        disced, scaling_factors[i,j], q_weights[i,j], tau = discretize_to_ternary(fp_weight[i,j])
            else: raise RuntimeError('Undefined conv_mode')
        else: raise RuntimeError('Undefined quant_mode')
        if quantize_inplace:
            layer.weight.data = scaling_factors * q_weights
    return scaling_factors.data, q_weights.data

# def discretize_to_binary(x: torch.Tensor, dim=-1)

class QuantLayer(nn.Module):
    """
    Module implementing a scaling-quantized linear layer
    """

    def __init__(self, original_layer, conv_mode='input_ch', quant_mode='binary'):
        """
        Create a quantized linear layer
        """
        super().__init__()
        self.original_layer = original_layer

        if isinstance(original_layer, (nn.Conv2d, nn.Linear)):
            scaling_factors, quant_weights = find_sfactors_qweights(original_layer, quant_mode=quant_mode, 
                conv_mode=conv_mode, quantize_inplace=False)
            self.scaling_factors = nn.Parameter(scaling_factors, requires_grad=True)
            self.quant_weights = nn.Parameter(quant_weights, requires_grad=False) # Avoid error: parameters on different devices
           
            # Note that here bias is not nn.Parameter
            if original_layer.bias is not None:
                self.bias = nn.Parameter(original_layer.bias.data.clone(), requires_grad=True)
            else:
                self.bias = None

    def forward(self, inputs: torch.Tensor):
        """
        Perform one forward pass through this layer.
        :param inputs:
        :return:
        """
        """
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
                            padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        """
        if isinstance(self.original_layer, (nn.Conv2d, nn.Linear)):
            new_weights = self.scaling_factors * self.quant_weights
            if isinstance(self.original_layer, nn.Conv2d):
                return nn.functional.conv2d(inputs, new_weights, self.original_layer.bias,
                                        self.original_layer.stride, self.original_layer.padding,
                                        self.original_layer.dilation, self.original_layer.groups)
                # torch.nn.functional.conv2d(input, weight, bias=None,
                #                           stride=1, padding=0, dilation=1, groups=1)
            elif isinstance(self.original_layer, nn.Linear):
                return nn.functional.linear(inputs, new_weights, self.bias)
                # torch.nn.functional.linear(input, weight, bias=None)
        else: # might help for future implementations of quantizing other layers 
            raise RuntimeWarning('a non-conv non-linear layer was a QuantLayer')
            return self.original_layer(inputs)

def replace_with_quantized(model, conv_mode='input_ch', exclude_first_last=True, quant_mode='binary'): 
    for name, child in model.named_children():
        if exclude_first_last and ((isinstance(child, nn.Conv2d) and child.in_channels == 3)
                                   or (isinstance(child, nn.Linear) and (child.out_features == 10 or child.out_features == 1000))):
            # only the first conv layer has input channel size = 3
            # only the last layer is linear
            continue
            
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            quant_layer = QuantLayer(child, conv_mode=conv_mode, quant_mode=quant_mode)
            setattr(model, name, quant_layer)
        else:
            replace_with_quantized(child, conv_mode, exclude_first_last, quant_mode)

## DELETE ALL ABOVE 
class QuantLayer(nn.Module):
    """
    Module implementing a scaling-quantized linear layer
    """

    def __init__(self, original_layer, conv_mode='input_ch', quant_mode='binary'):
        """
        Create a quantized linear layer
        """
        super().__init__()
        self.original_layer = original_layer

        if isinstance(original_layer, (nn.Conv2d, nn.Linear)):
            scaling_factors, quant_weights = find_sfactors_qweights(original_layer, quant_mode=quant_mode, 
                conv_mode=conv_mode, quantize_inplace=False)
            self.scaling_factors = nn.Parameter(scaling_factors, requires_grad=True)
            self.quant_weights = nn.Parameter(quant_weights, requires_grad=False) # Avoid error: parameters on different devices
           
            # Note that here bias is not nn.Parameter
            if original_layer.bias is not None:
                self.bias = nn.Parameter(original_layer.bias.data.clone(), requires_grad=True)
            else:
                self.bias = None

    def forward(self, inputs: torch.Tensor):
        """
        Perform one forward pass through this layer.
        :param inputs:
        :return:
        """
        """
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
                            padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        """
        if isinstance(self.original_layer, (nn.Conv2d, nn.Linear)):
            new_weights = self.scaling_factors * self.quant_weights
            if isinstance(self.original_layer, nn.Conv2d):
                return nn.functional.conv2d(inputs, new_weights, self.original_layer.bias,
                                        self.original_layer.stride, self.original_layer.padding,
                                        self.original_layer.dilation, self.original_layer.groups)
                # torch.nn.functional.conv2d(input, weight, bias=None,
                #                           stride=1, padding=0, dilation=1, groups=1)
            elif isinstance(self.original_layer, nn.Linear):
                return nn.functional.linear(inputs, new_weights, self.bias)
                # torch.nn.functional.linear(input, weight, bias=None)
        else: # might help for future implementations of quantizing other layers 
            raise RuntimeWarning('a non-conv non-linear layer was a QuantLayer')
            return self.original_layer(inputs)
def replace_with_Bbit(model, conv_mode='input_ch', exclude_first_last=True, quant_mode='binary'): 
    for name, child in model.named_children():
        if exclude_first_last and ((isinstance(child, nn.Conv2d) and child.in_channels == 3)
                                   or (isinstance(child, nn.Linear) and (child.out_features == 10 or child.out_features == 1000))):
            # only the first conv layer has input channel size = 3
            # only the last layer is linear
            continue
            
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            quant_layer = BbitLayer(child, conv_mode=conv_mode, quant_mode=quant_mode)
            setattr(model, name, quant_layer)
        else:
            replace_with_quantized(child, conv_mode, exclude_first_last, quant_mode)        
      