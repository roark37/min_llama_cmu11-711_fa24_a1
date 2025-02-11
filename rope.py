from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    # # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...
    # query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1) 
    # key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    
    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    theta_seq = torch.tensor([theta ** (-2*(i-1)/head_dim) for i in range(1, head_dim // 2 + 1)]) # shape: (head_dim/2, )
    t = torch.arange(max_seq_len).unsqueeze(dim=1)  # t shape: (max_seq_len, 1)
    theta_t_seq = t * theta_seq  # broadcast, shape (max_seq_len, head_dim/2)
    cos_real = torch.cos(theta_t_seq).unsqueeze(dim=0)  # shape (1, max_seq_len, head_dim/2)
    sin_imag = torch.sin(theta_t_seq).unsqueeze(dim=0)  # shape (1, max_seq_len, head_dim/2)
    cos_sin = torch.cat((cos_real, sin_imag), dim=0).permute(1, 2, 0) # shape (max_seq_len, head_dim/2, 2)

    angle_cut = cos_sin[:seqlen][None, None, ...].contiguous()   # shape (1, 1, T, head_dim/2, 2)
    angle_complex_view = torch.view_as_complex(angle_cut)
    
    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.

    q_seq = query.float().reshape(query.shape[:-1] + (-1, 2)) # shape: (B, T, localH, head_dim/2, 2)
    q_seq = q_seq.transpose(1, 2).contiguous()                # shape: (B, localH, T, head_dim/2, 2)
    q_complex_view = torch.view_as_complex(q_seq)
    query_out = torch.view_as_real(q_complex_view * angle_complex_view).reshape(q_seq.shape[:-2] + (-1,))
                                                              # shape: (B, localH, T, head_dim)
    query_out = query_out.transpose(1, 2)                     # shape: (B, T, localH, head_dim)

    k_seq = key.float().reshape(query.shape[:-1] + (-1, 2)) # shape: (B, T, localH, head_dim/2, 2)
    k_seq = k_seq.transpose(1, 2).contiguous()              # shape: (B, localH, T, head_dim/2, 2)    
    k_complex_view = torch.view_as_complex(k_seq)
    key_out = torch.view_as_real(k_complex_view * angle_complex_view).reshape(q_seq.shape[:-2] + (-1,))
                                                            # shape: (B, localH, T, head_dim)
    key_out = key_out.transpose(1, 2)                       # shape: (B, T, localH, head_dim)
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out