"""
Code for getting intermediate features.
"""

import torch
import math

# add hook
# hookDict = {}
# def getFeatures(name):
#     # the hook signature
#     def hook(model, input, output):
#         hookDict[name] = output
#     return hook
# h1 = self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.register_forward_hook(getFeatures("feat"))

# attn_text, attn_image, attn_uncond = store_processor.attention_probs.chunk(3)
# bh, hw1, hw2 = attn_uncond.shape
# h = list(self.unet.config.attention_head_dim)[-1]
# features = attn_uncond.reshape(batch_size, h, hw1, hw2)

# feat_text, feat_image, feat_uncond = store_processor.features.chunk(3)
# bh, hw1, hw2 = feat_uncond.shape
# h = list(self.unet.config.attention_head_dim)[-1]
# features = feat_uncond.reshape(batch_size, h, hw1, hw2)

# detach the hooks
# # h1.remove()


# class AttnProcessor2_0:
#     r"""
#     Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
#     """

#     def __init__(self):
#         if not hasattr(F, "scaled_dot_product_attention"):
#             raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states,
#         encoder_hidden_states=None,
#         attention_mask=None,
#         temb=None,
#         scale: float = 1.0,
#     ):
#         residual = hidden_states

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#         batch_size, sequence_length, _ = (
#             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#         )

#         if attention_mask is not None:
#             attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
#             # scaled_dot_product_attention expects attention_mask shape to be
#             # (batch, heads, source_length, target_length)
#             attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states, lora_scale=scale)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         key = attn.to_k(encoder_hidden_states, lora_scale=scale)
#         value = attn.to_v(encoder_hidden_states, lora_scale=scale)

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads

#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )

#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states, lora_scale=scale)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor


# processes and stores attention probabilities
class CrossAttnStoreProcessor:
    """
    From https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_sag.py
    """

    def __init__(self):
        self.attention_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def get_self_attn(self, latents_shape, h, show=False):
        # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
        attn_text, attn_image, attn_uncond = self.attention_probs.chunk(3)
        attn_map = attn_image

        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = latents_shape

        # Produce attention mask
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_map = attn_map.mean(1, keepdim=False).sum(1, keepdim=False)
        size = int(math.sqrt(attn_map.shape[-1]))
        attn_map = attn_map.reshape(b, size, size).unsqueeze(1)

        attn_map = torch.nn.functional.interpolate(attn_map, (latent_h, latent_w))

        if show:
            import mediapy

            mediapy.show_images(attn_map.permute(0, 2, 3, 1).detach().cpu(), height=200)

        return attn_map
