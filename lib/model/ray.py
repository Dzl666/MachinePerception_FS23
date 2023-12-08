import torch

def cumsum(x, exclusive=True):
    """Computes the cumulative sum of the elements of x along the last dimension.
    Args:
        x (torch.Tensor): tensor of shape :math:`[..., Ns, 1]`.
        exclusive (bool): whether to compute exclusive cumsum or not. (default: True)
    Returns:
        torch.Tensor: tensor of shape :math:``[..., Ns, 1]`.
    """
    x = x.squeeze(-1)
    if exclusive:
        c = torch.cumsum(x, dim=-1)
        # "Roll" the elements along dimension 'dim' by 1 element.
        c = torch.roll(c, 1, dims=-1)
        # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
        c[..., 0] = 0.0
        return c[..., None]
    else:
        return torch.cumsum(x, dim=-1)[..., None]

# Exponential transmittance is derived from the Beer-Lambert law. Typical implementations is calculated 
# with :func:`cumprod`, but the exponential allows a reformulation as a :func:`cumsum` which its gradient
# is more stable and faster to compute. We opt to use the :func:`cumsum` formulation.
# For more details, we recommend "Monte Carlo Methods for Volumetric Light Transport" by Novak et al.

def exponential_integration(feats, tau, depth, exclusive=True):
    """Exponential transmittance integration across packs using the optical thickness (tau).
    Args (torch.FloatTensor):
        feats: features of shape [..., Num_sample, num_feats].
        tau: optical thickness of shape [..., Ns, 1].
        depth: depth of shape [..., Ns, 1].
        exclusive (bool): Compute exclusive exponential integration if true. (default: True)
    Returns (torch.FloatTensor):
        feats_out: Integrated features of shape [..., num_feats]
        depth_out: Integrated depth of shape [..., 1]
        weight_sum: sum of all weight along a ray [..., 1]
        weights: Weights of shape [..., Ns]
    """
    # TODO(ttakikawa): This should be a fused kernel... we're iterating over packs, so might as well also perform the integration in the same manner.
    
    # calculate termination probability
    transmittance = torch.exp(-1.0 * cumsum(tau.contiguous(), exclusive=exclusive))
    alpha = 1.0 - torch.exp(-tau.contiguous())
    
    weights = transmittance * alpha # [B, Nr, Ns, 1]
    # weights = torch.nn.functional.normalize(weights, dim=-2)
    feats_out = torch.sum(weights * feats, dim=-2)
    depth_out = torch.sum(weights * depth, dim=-2)
    weight_sum = torch.sum(weights, dim=-2)

    return feats_out, depth_out, weight_sum, weights.squeeze(-1)
