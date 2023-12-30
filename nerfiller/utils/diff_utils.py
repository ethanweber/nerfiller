"""
Diffusion pipeline utils.
"""

import torch


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


# VAE decoder approximation as specified in Latent-NeRF https://arxiv.org/pdf/2211.07600.pdf
# and this blog post https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204
def get_decoder_approximation():
    return torch.tensor(
        [
            [0.298, 0.207, 0.208],
            [0.187, 0.286, 0.173],
            [-0.158, 0.189, 0.264],
            [-0.184, -0.271, -0.473],
        ]
    )


def get_epsilon_from_v_prediction(
    v_prediction: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    alphas_cumprod,
):
    """Returns epsilon from v_prediction.

    Args:
        v_prediction: The predicted velocity.
        timestep: The current discrete timestep in the diffusion chain.
        sample: A current instance of a sample created by the diffusion process.
        alphas_cumprod: TODO
    """

    alpha_prod_t = alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    return (alpha_prod_t**0.5) * v_prediction + (beta_prod_t**0.5) * sample


def get_v_prediction_from_epsilon(
    pred_epsilon: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    alphas_cumprod,
):
    """Returns epsilon from v_prediction.

    Args:
        pred_epsilon:
        timestep: The current discrete timestep in the diffusion chain.
        sample: A current instance of a sample created by the diffusion process.
        alphas_cumprod: TODO
    """

    alpha_prod_t = alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    v_prediction = (pred_epsilon - (beta_prod_t**0.5) * sample) / (alpha_prod_t**0.5)
    return v_prediction


def register_extended_attention(unet):
    """Method from Tune-A-Video, but code modified from TokenFlow codebase."""

    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            # Here we are making an assumption about passing in 3 varients of conditioning into the model
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            k_0 = k[:n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_1 = k[n_frames : 2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_2 = k[2 * n_frames :].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_0 = v[:n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_1 = v[n_frames : 2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_2 = v[2 * n_frames :].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            q_0 = self.head_to_batch_dim(q[:n_frames])
            q_1 = self.head_to_batch_dim(q[n_frames : 2 * n_frames])
            q_2 = self.head_to_batch_dim(q[2 * n_frames :])
            k_0 = self.head_to_batch_dim(k_0)
            k_1 = self.head_to_batch_dim(k_1)
            k_2 = self.head_to_batch_dim(k_2)
            v_0 = self.head_to_batch_dim(v_0)
            v_1 = self.head_to_batch_dim(v_1)
            v_2 = self.head_to_batch_dim(v_2)

            out_0 = []
            out_1 = []
            out_2 = []

            q_0 = q_0.view(n_frames, h, sequence_length, dim // h)
            k_0 = k_0.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_0 = v_0.view(n_frames, h, sequence_length * n_frames, dim // h)
            q_1 = q_1.view(n_frames, h, sequence_length, dim // h)
            k_1 = k_1.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_1 = v_1.view(n_frames, h, sequence_length * n_frames, dim // h)
            q_2 = q_2.view(n_frames, h, sequence_length, dim // h)
            k_2 = k_2.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_2 = v_2.view(n_frames, h, sequence_length * n_frames, dim // h)

            for j in range(h):
                sim_0 = torch.bmm(q_0[:, j], k_0[:, j].transpose(-1, -2)) * self.scale
                sim_1 = torch.bmm(q_1[:, j], k_1[:, j].transpose(-1, -2)) * self.scale
                sim_2 = torch.bmm(q_2[:, j], k_2[:, j].transpose(-1, -2)) * self.scale

                out_0.append(torch.bmm(sim_0.softmax(dim=-1), v_0[:, j]))
                out_1.append(torch.bmm(sim_1.softmax(dim=-1), v_1[:, j]))
                out_2.append(torch.bmm(sim_2.softmax(dim=-1), v_2[:, j]))

            out_0 = (
                torch.cat(out_0, dim=0)
                .view(h, n_frames, sequence_length, dim // h)
                .permute(1, 0, 2, 3)
                .reshape(h * n_frames, sequence_length, -1)
            )
            out_1 = (
                torch.cat(out_1, dim=0)
                .view(h, n_frames, sequence_length, dim // h)
                .permute(1, 0, 2, 3)
                .reshape(h * n_frames, sequence_length, -1)
            )
            out_2 = (
                torch.cat(out_2, dim=0)
                .view(h, n_frames, sequence_length, dim // h)
                .permute(1, 0, 2, 3)
                .reshape(h * n_frames, sequence_length, -1)
            )

            out = torch.cat([out_0, out_1, out_2], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    for _, unet_module in unet.named_modules():
        if isinstance_str(unet_module, "BasicTransformerBlock"):
            module = unet_module.attn1
            module.forward = sa_forward(module)

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
