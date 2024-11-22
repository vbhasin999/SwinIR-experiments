import torch
import torch.nn.utils.prune as prune
import os
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

##############################################################################
#                   SPARSITY CALCULATION FUNCTIONS
##############################################################################

def calculate_sparsity(param):
    """
    Calculate the sparsity of a given tensor.

    Sparsity is defined as the proportion of zero elements in the tensor.

    Args:
        param (torch.Tensor): The input tensor for which sparsity is to be calculated.

    Returns:
        float: The sparsity of the tensor, a value between 0 and 1 where 0 means no sparsity (all elements are non-zero) and 1 means full sparsity (all elements are zero).
    """
    nonzero = param[param != 0].numel()
    total = param.numel()
    return 1 - (nonzero / total)

def calculate_sparsity_overall(params_to_prune):
    """
    Calculate the overall sparsity of the given parameters.
    Sparsity is defined as the proportion of zero-valued elements in the parameters.
    Args:
        params_to_prune (list of tuples): A list where each tuple contains a module and the name of the parameter to be pruned.
    Returns:
        float: The overall sparsity of the parameters, a value between 0 and 1.
    """

    total_nonzero = 0
    total_params = 0
    for module, param_name in params_to_prune:
        param = getattr(module, param_name)
        total_nonzero += param[param != 0].numel()
        total_params += param.numel()
    return 1 - (total_nonzero / total_params)

def calculate_sparsity_model(model):
    """
    Calculate the sparsity of a given model.
    Sparsity is defined as the proportion of zero-valued parameters in the model.
    Args:
        model (torch.nn.Module): The model for which to calculate sparsity.
    Returns:
        float: The sparsity of the model, a value between 0 and 1.
    """

    total_nonzero = sum(p[p != 0].numel() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    return 1 - (total_nonzero / total_params)

def calculate_sparse_model_size(model, temp_file=None):
    """
    Calculate the size of a sparse model in megabytes.
    This function removes the reparameterization from the model's weights and biases,
    converts the model's state dictionary to a sparse format, saves it to a file,
    and measures the file size in megabytes.
    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        temp_file (str, optional): The path to the temporary file where the sparse model
                                   will be saved. If None, a default file name "temp_sparse_model.pt"
                                   will be used.
    Returns:
        float: The size of the sparse model in megabytes.
    """


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

    return size_mb


##############################################################################
#                   ATTENTION HEAD PRUNING FUNCTIONS
##############################################################################
def prune_attn_heads_layer(param, num_heads_to_prune, plot=False):
    """
    Prunes the specified number of attention heads from a given parameter tensor.
    Args:
        param (torch.nn.Module): The parameter tensor containing the weights of the attention heads.
        num_heads_to_prune (int): The number of attention heads to prune.
        plot (bool, optional): If True, plots the attention heads before and after pruning. Default is False.
    Returns:
        None: The function modifies the input parameter tensor in place.
    Notes:
        - The function assumes that the input parameter tensor has a shape that can be reshaped into 
          (3, num_heads, head_dim, inp_dim), where `3` corresponds to the query, key, and value weights.
        - The pruning is based on the L1 norm of the weights of the attention heads.
        - The heads with the smallest L1 norms are pruned by zero
    """
    # Ensure param is contiguous for reshaping
    num_heads = param.num_heads
    weight = param.qkv.weight.contiguous()
    out, inp = weight.shape

    # param is linear layer weight of shape (out_features, in_features)
    # since we know that out_features = 3 * in_features, we can reshape this to 3, self.num_heads, C // self.num_heads, inp

    # Reshape to (3, num_heads, head_dim, inp_dim)
    head_dim = out // (3 * num_heads)
    reshaped_weight = weight.reshape(3, num_heads, head_dim, inp)

    if plot: plot_attention_heads(reshaped_weight, title="Before Pruning")

    l1_norms = torch.norm(reshaped_weight, p=1, dim=(2, 3)) # shape: (3, num_heads)

    avg_l1_norms = l1_norms.mean(dim=0)  # Shape: (num_heads,) average each head over q,k,v

    heads_to_prune = torch.argsort(avg_l1_norms)[:num_heads_to_prune]

    # Zero out the weights of the selected heads
    mask = torch.ones_like(reshaped_weight)
    mask[:, heads_to_prune, :, :] = 0  # Set weights of selected heads to zero
    pruned_weight = reshaped_weight * mask
    if plot: plot_attention_heads(pruned_weight, title="After Pruning")

    # Reshape back and assign the modified weights
    param.qkv.weight = torch.nn.Parameter(pruned_weight.reshape(out, inp))



def prune_attn_heads(model, heads_to_prune, plot_freq=10):
    """
    Prunes attention heads in the given model.
    Args:
        model (torch.nn.Module): The model containing attention layers to prune.
        heads_to_prune (int): The number of attention heads to prune.
        plot_freq (int, optional): Frequency of plotting the pruning process. Default is 10.
    Returns:
        None
    This function iterates over the modules in the model, identifies those with attention layers,
    and prunes the specified number of attention heads. It also calculates and prints the overall
    sparsity of the pruned model and the size of the sparse model.
    """

    params_to_prune = []
    layers_processed = 0

    for module in model.modules():
        if hasattr(module, 'attn'):
            # print(f'Pruning {heads_to_prune} heads in {module.attn}')
            plot = (layers_processed + 1) % plot_freq == 0
            params_to_prune.append((module.attn.qkv, "weight"))
            prune_attn_heads_layer(module.attn, heads_to_prune, plot)
            layers_processed += 1
    
    print(f'Overall sparsity: {calculate_sparsity_overall(params_to_prune):.4f}')
    print(f'sparse model size: {calculate_sparse_model_size(model, temp_file=f"pruned_models/structured_attn_heads_{heads_to_prune}.pt"):.4f}')


def plot_attention_heads(attn_weights, head_dim=1, title="Attention Heads"):
    """
    Plots attention heatmaps for each attention head.

    Args:
        attn_weights (torch.Tensor or np.ndarray): Attention weights tensor of shape 
                                                   (batch_size, num_heads, seq_len, seq_len).
        head_dim (int): Dimension representing the attention heads (default: 1).
        title (str): Title for the plot.

    Returns:
        None: Displays the heatmaps.
    """
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()

    # Ensure attn_weights is 4D: (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.ndim == 4, "Expected attention weights of shape (3, num_heads, seq_len, seq_len)."

    qkv, num_heads, seq_len, _ = attn_weights.shape

    # Plot heatmaps for all heads in the first batch
    fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))
    fig.suptitle(title, fontsize=16)

    for head_idx in range(num_heads):
        ax = axes[head_idx] if num_heads > 1 else axes
        sns.heatmap(
            attn_weights[0, head_idx],
            ax=ax,
            cmap="icefire",
            cbar=True,
            xticklabels=False,
            yticklabels=False
        )
        ax.set_title(f"Head {head_idx + 1}")

    plt.tight_layout()
    plt.show()


##############################################################################
#                       LINEAR LAYER PRUNING FUNCTIONS
##############################################################################

def prune_linear_nodes(model, num_nodes_to_prune, plot_freq=50):
    """
    Prunes a specified number of nodes from the linear layers (fully connected layers) of a given model.
    Args:
        model (torch.nn.Module): The neural network model containing the layers to be pruned.
        num_nodes_to_prune (int): The number of nodes to prune from each linear layer.
        plot_freq (int, optional): Frequency of plotting the weights before and after pruning. Default is 50.
    Returns:
        None
    This function iterates through the modules of the given model, identifies the linear layers (fully connected layers)
    within the 'mlp' attribute, and prunes the specified number of nodes from the weights of these layers. It also plots
    the weights before and after pruning at the specified frequency. Finally, it prints the overall sparsity of the model
    and the size of the sparse model.
    Note:
        - The function assumes that the model has an 'mlp' attribute containing the linear layers to be pruned.
        - The function uses structured pruning with L1 norm along the specified dimension (dim=1).
    """
    
    params_to_prune = []
    layers_processed = 0

    for module in model.modules():
        if hasattr(module, 'mlp'):
            for submodule_name, submodule in module.mlp.named_children():
                if submodule_name.startswith('fc'):
                    plot = (layers_processed + 1) % plot_freq == 0
                    params_to_prune.append((submodule, "weight"))

                    if plot: plot_linear_weights(submodule, title=f"Layer {layers_processed} Before Pruning {submodule_name}")
                    prune.ln_structured(submodule, name="weight", amount=num_nodes_to_prune, n=1, dim=1)
                    if plot: plot_linear_weights(submodule, title=f"Layer {layers_processed} After Pruning {submodule_name}")

                    layers_processed += 1
    
    print(f'Overall sparsity: {calculate_sparsity_overall(params_to_prune):.4f}')
    print(f'sparse model size: {calculate_sparse_model_size(model, temp_file=f"pruned_models/structured_fc_neurons_{num_nodes_to_prune}.pt"):.4f}')


def plot_linear_weights(layer, title="Linear Layer Weights", cmap="icefire"):
    """
    Plots a heatmap of the weights of a given linear layer.

    Args:
        layer (torch.nn.Linear): The linear layer whose weights are to be visualized.
        title (str): Title for the plot.
        cmap (str): Colormap to use for the heatmap (default: 'viridis').

    Raises:
        ValueError: If the layer is not an instance of torch.nn.Linear.
    """
    if not isinstance(layer, torch.nn.Linear):
        raise ValueError("The provided layer is not a torch.nn.Linear layer.")

    if hasattr(layer, 'weight_orig'):
            prune.remove(layer, 'weight') 
    # Get the weights and convert them to a NumPy array
    weights = layer.weight.detach().cpu().numpy()

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(weights, cmap=cmap, cbar=True, annot=False)
    plt.title(title, fontsize=16)
    plt.xlabel("Input Features", fontsize=12)
    plt.ylabel("Output Features", fontsize=12)
    plt.tight_layout()
    plt.show()