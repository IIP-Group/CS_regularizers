import torch

def lplq_regularization_CNN(model, op_mode='train', conv_regu='binary', fc_regu='binary',
                            normalization_type='elenum', p=4, q=2, r=2,
                            exclude_first_last=True):
    # regularize conv.weight by kernel, regularize fc.weight by row
    lplq_reg = 0.0  # Initialization, if no operation, return 0
    marker = 0  # Initialization
    normalizing_factor = 1  # Initialization
    if 1 <= q < p:
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue
            if exclude_first_last and ((param.dim() == 4 and param.shape[1] == 3) 
                                        or (param.dim() == 2 and (param.shape[0] == 10 
                                                                    or param.shape[0] == 1000))):
                continue
               
            if param.dim() == 4:  # conv.weight regularization by kernel
                if param.numel() == 0:  # conv binary
                    continue

                if conv_regu == 'binary':
                    param_reshaped = param.reshape(param.shape[0], param.shape[1], -1)  # Todo: verify adding .data
                    if normalization_type == 'zero_norm':
                        normalizing_factor = len(param_reshaped.nonzero())  # binary should not use zero_norm,
                        #  because zero have to quantize to -1 or +1
                    elif normalization_type == 'elenum':
                        normalizing_factor = param_reshaped.numel()
                    elif normalization_type == 'none':
                        normalizing_factor = 1
                    else:
                        raise RuntimeError('normalization_type should be \'zero_norm\', \'elenum\' , or \'none\'. Default is zero_norm')
                    lplq_reg += ((param_reshaped.shape[-1]) ** (r / q - r / p) * torch.sum(
                        torch.linalg.norm(param_reshaped, ord=p, dim=2) ** r) - torch.sum(
                        torch.linalg.norm(param_reshaped, ord=q, dim=2) ** r)) / normalizing_factor
                elif conv_regu == 'binary_input_ch':
                    param_reshaped = param.reshape(param.shape[0], -1)
                    if normalization_type == 'zero_norm':
                        normalizing_factor = len(param_reshaped.nonzero())
                    elif normalization_type == 'elenum':
                        normalizing_factor = param_reshaped.numel()
                    elif normalization_type == 'none':
                        normalizing_factor = 1
                    else:
                        raise RuntimeError('normalization_type should be \'zero_norm\', \'elenum\' , or \'none\'. Default is zero_norm')
                    lplq_reg += ((param_reshaped.shape[-1])**(r / q - r / p) * torch.sum(
                        torch.linalg.norm(param_reshaped, ord=p, dim=1) ** r
                    ) - torch.sum(torch.linalg.norm(param_reshaped, ord=q, dim=1)** r))  / normalizing_factor
                elif conv_regu == 'ternary':
                    param_reshaped = param.reshape(param.shape[0], param.shape[1], -1)  # Todo: verify adding .data
                    if normalization_type == 'zero_norm':
                        normalizing_factor = len(param_reshaped.nonzero())  # binary should not use zero_norm,
                        #  because zero have to quantize to -1 or +1
                    elif normalization_type == 'elenum':
                        normalizing_factor = param_reshaped.numel()
                    elif normalization_type == 'none':
                        normalizing_factor = 1
                    else:
                        raise RuntimeError('normalization_type should be \'zero_norm\', \'elenum\' , or \'none\'. Default is zero_norm')
                    x = param_reshaped
                    lplq_reg += torch.sum((torch.linalg.norm(x, ord=2, dim=2)**2 * torch.linalg.norm(x, ord=6, dim=2)**6
                                - torch.linalg.norm(x, ord=4, dim=2)**8)) / normalizing_factor
                elif conv_regu == 'ternary_input_ch': 
                    param_reshaped = param.reshape(param.shape[0], -1)  # Todo: verify adding .data
                    if normalization_type == 'zero_norm':
                        normalizing_factor = len(param_reshaped.nonzero())  # binary should not use zero_norm,
                        #  because zero have to quantize to -1 or +1
                    elif normalization_type == 'elenum':
                        normalizing_factor = param_reshaped.numel()
                    elif normalization_type == 'none':
                        normalizing_factor = 1
                    else:
                        raise RuntimeError('normalization_type should be \'zero_norm\', \'elenum\' , or \'none\'. Default is zero_norm')
                    x = param_reshaped
                    # lplq_reg += torch.sum((torch.linalg.norm(x, ord=2, dim=1)**2 * torch.linalg.norm(x, ord=6, dim=1)**6
                    #             - torch.linalg.norm(x, ord=4, dim=1)**8)) / normalizing_factor
                    lplq_reg += torch.sum(torch.sum(x**2, dim=1) * torch.sum(x**6, dim=1)
                                - torch.sum(x**4, dim=1)**2) / normalizing_factor

                elif conv_regu == 'none':
                    lplq_reg += 0
                else:
                    raise RuntimeError("The \'conv_regu\' should be set as binary, ternary or none.")
            elif param.dim() == 2:  # fc.weight regularization by row
                assert 'fc' in name or 'classifier' in name
                if param.numel() == 0:  # fc binary
                    continue

                if fc_regu == 'binary':
                    # 加.data后出现问题, 参数分布就没有受到影响, Todo 考虑一下为什么
                    # Select normalization type
                    if normalization_type == 'zero_norm':
                        normalizing_factor = len(param.nonzero())
                    elif normalization_type == 'elenum':
                        normalizing_factor = param.numel()
                    elif normalization_type == 'none':
                        normalizing_factor = 1
                    else:
                        raise RuntimeError('normalization_type should be \'zero_norm\', \'elenum\' , or \'none\'. Default is zero_norm')

                    ################################# DEBUG BLOCK #######################################
                    # lplq_temp = (param.shape[1] ** (r / q - r / p) * torch.sum(
                    #     torch.linalg.norm(param, ord=p, dim=1) ** r) - torch.sum(
                    #     torch.linalg.norm(param, ord=q, dim=1) ** r)) / normalizing_factor
                    # if lplq_temp < - 1e-5:  # 这里的lplq_temp小于0, 应该是因为量化误差
                    #     print('breakpoint: negative lplq for ternary', name, p, q, r)
                    #     print(param.shape[0])
                    #     print(param.shape[1])
                    #     print(param.shape[0] ** (r / q - r / p))
                    #     print(torch.linalg.norm(param, ord=p, dim=1))
                    #     print(torch.sum(torch.linalg.norm(param, ord=p, dim=1) ** r))
                    #     print(
                    #         param.shape[1] ** (r / q - r / p) * torch.sum(torch.linalg.norm(param, ord=p, dim=1) ** r))
                    #     print(torch.linalg.norm(param, ord=q, dim=1))
                    #     print(torch.linalg.norm(param, ord=q, dim=1) ** r)
                    #     print(torch.sum(torch.linalg.norm(param, ord=q, dim=1) ** r))
                    #     print(lplq_temp * normalizing_factor)
                    #     print(param.shape[1] ** (r / q - r / p) * torch.sum(
                    #         torch.linalg.norm(param, ord=p, dim=1) ** r) - torch.sum(
                    #         torch.linalg.norm(param, ord=q, dim=1) ** r))
                    #######################################################################################

                    lplq_reg += (param.shape[1] ** (r / q - r / p) * torch.sum(
                        torch.linalg.norm(param, ord=p, dim=1) ** r) - torch.sum(
                        torch.linalg.norm(param, ord=q, dim=1) ** r)) / normalizing_factor
                elif fc_regu == 'ternary':
                    if normalization_type == 'zero_norm':
                        normalizing_factor = len(param_reshaped.nonzero())  # binary should not use zero_norm,
                        #  because zero have to quantize to -1 or +1
                    elif normalization_type == 'elenum':
                        normalizing_factor = param_reshaped.numel()
                    elif normalization_type == 'none':
                        normalizing_factor = 1
                    else:
                        raise RuntimeError('normalization_type should be \'zero_norm\', \'elenum\' , or \'none\'. Default is zero_norm')
                    x = param_reshaped
                    lplq_reg += torch.sum((torch.linalg.norm(x, ord=2, dim=1)**2 * torch.linalg.norm(x, ord=6, dim=1)**6
                                - torch.linalg.norm(x, ord=4, dim=1)**8)) / normalizing_factor
                    
                elif fc_regu == 'none':
                    lplq_reg += 0
                else:
                    raise RuntimeError("The \'fc_regu\' should be set as binary, ternary, or none.")
            # else:
            #     # print(name, param.shape)
            #     continue
    else:
        raise RuntimeError('Lq_Lq regularization requires 1 <= q < p ')
    return lplq_reg



