import torch
import torch.nn as nn


def wrapper(module: nn.Module, x, seq_dim: bool):
    """Helper to handle sequential/non-sequential input for RNN and MLP modules.
    
    Args:
        module: The nn.Module to apply (RNN or MLP)
        x: Input tensor, or tuple (input, hidden_states) for RNN
        seq_dim: If True, input has sequence dimension (batch mode training)
                 If False, input is single timestep (inference mode)
    
    Returns:
        For RNN: (output, hidden_states)
        For MLP: output tensor
    """
    if isinstance(module, nn.RNNBase):
        x, hidden = x
        if seq_dim:
            return module(x, hidden)
        else:
            out, hidden = module(x.unsqueeze(0), hidden)
            return out.squeeze(0), hidden
    elif seq_dim:
        # MLP with sequence: flatten seq and batch, then unflatten
        n_seq = x.size(0)
        return module(x.flatten(0, 1)).unflatten(0, (n_seq, x.size(1)))
    else:
        return module(x)

