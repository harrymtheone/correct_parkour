import torch

@torch.jit.script
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = x.flatten(start_dim=2).float()

    # Prevent empty selection (avoid NaN due to division by zero)
    if not mask.any():
        return torch.tensor(0.0, device=x.device)

    return x[mask].mean()


@torch.jit.script
def masked_MSE(input_: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Flatten everything
    input_f = input_.flatten(start_dim=2).float()
    target_f = target.flatten(start_dim=2).float()

    # Select masked elements
    input_sel = input_f[mask]
    target_sel = target_f[mask]

    # Prevent empty selection (avoid NaN due to division by zero)
    if not mask.any():
        return torch.tensor(0.0, device=input_.device)

    # Compute MSE on selected elements
    return (input_sel - target_sel).square().mean()


@torch.jit.script
def masked_L1(input_: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Flatten
    input_f = input_.flatten(start_dim=2).float()
    target_f = target.flatten(start_dim=2).float()

    # Select masked values
    input_sel = input_f[mask]
    target_sel = target_f[mask]

    # Prevent empty selection (avoid NaN due to division by zero)
    if not mask.any():
        return torch.tensor(0.0, device=input_.device)

    # L1 loss
    return (input_sel - target_sel).abs().mean()


@torch.jit.script
def masked_vae_kl_loss(mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the VAE KL Divergence loss considering only valid elements.
    Calculations are performed in float32.
    """
    mu_f = mu.float()
    logvar_f = logvar.float()

    # Clamp logvar to prevent exp() overflow
    # std range: [0.0067, 2.72] which is reasonable for most applications
    logvar_clamped = torch.clamp(logvar_f, min=-10.0, max=2.0)

    # Compute KL divergence element-wise: KL(N(mu, sigma^2) || N(0, 1))
    # = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
    kl_element = 1.0 + logvar_clamped - mu_f.square() - logvar_clamped.exp()

    # Mask out invalid regions before summation
    # We must expand mask to match kl_element shape if necessary, assuming mask is [Batch, Seq] or [Batch, Seq, 1]
    # and kl_element is [Batch, Seq, LatentDim]
    # NOTE: masked_mean expects mask to align with input after flatten(start_dim=2),
    # but here we sum over dim=2 first.
    
    # Expand mask to [Batch, Seq, LatentDim] to zero out invalid KL terms
    if mask.dim() < kl_element.dim():
        mask_expanded = mask.expand(kl_element.shape)
    else:
        mask_expanded = mask
        
    masked_kl_element = torch.where(mask_expanded, kl_element, torch.zeros_like(kl_element))

    # Sum over latent dimension (dim=2)
    # result shape: [Batch, Seq, 1]
    kl_sum = masked_kl_element.sum(dim=2, keepdim=True)
    
    # masked_mean will flatten from dim=2, so [Batch, Seq, 1] becomes [Batch, Seq * 1] -> [Batch, Seq] effectively
    # and mask is likely [Batch, Seq] or [Batch, Seq, 1]
    return -0.5 * masked_mean(kl_sum, mask)
