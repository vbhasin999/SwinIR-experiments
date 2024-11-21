import torch
import torch.nn.utils.prune as prune
import os

def calculate_sparsity(param):
    nonzero = param[param != 0].numel()
    total = param.numel()
    return 1 - (nonzero / total)

def calculate_sparsity_overall(model, params_to_prune):
    total_nonzero = 0
    total_params = 0
    for module, param_name in params_to_prune:
        param = getattr(module, param_name)
        total_nonzero += param[param != 0].numel()
        total_params += param.numel()
    return 1 - (total_nonzero / total_params)

def calculate_sparsity_model(model):
    total_nonzero = sum(p[p != 0].numel() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    return 1 - (total_nonzero / total_params)

# def prune_model(model, sparsity, mode="all"):
#     for module, param_name in params_to_prune:
#         prune.l1_unstructured(module, name=param_name, amount=sparsity)
#     return model

def calculate_sparse_model_size(model, temp_file=None):
    from copy import deepcopy

    for module in model.modules():
        if hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')  # Remove reparameterization
        if hasattr(module, 'bias_orig'):
            prune.remove(module, 'bias')  # Remove bias reparameterization if applicable
    
    print(f'model sparsity: {calculate_sparsity_model(model):.4f}')

    # Convert the state dict to sparse format
    sparse_state_dict = model.state_dict()
    for key in sparse_state_dict:
        sparse_state_dict[key] = sparse_state_dict[key].to_sparse()

    # Save the sparse model to disk and measure size
    if temp_file is None:
        temp_file = "temp_sparse_model.pt"

    torch.save(sparse_state_dict, temp_file)
    size_mb = os.path.getsize(temp_file) / 1e6
    # os.remove(temp_file)  # Clean up temporary file
    return size_mb