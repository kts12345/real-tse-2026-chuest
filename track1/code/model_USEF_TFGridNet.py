import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import copy

from models.local.PositionalEncoding import PositionalEncoding
from models.local.TFgridnet import GridNetV2Block

EPS = 1e-8


class _CausalPerFrameGroupNorm(nn.Module):
    """Causal drop-in replacement for ``nn.GroupNorm(1, num_channels)`` applied
    to a ``[B, C, T, Q]`` tensor.

    The original GroupNorm(1, C) computes statistics jointly over (C, T, Q),
    i.e. it mixes information across the *whole* time axis -> it peeks into
    future frames. This version computes the mean/variance independently for
    each time frame over (C, Q) only, so frame ``t`` never depends on ``t'>t``.

    The affine parameters keep the exact name/shape of ``nn.GroupNorm`` --
    ``weight`` and ``bias`` of shape ``[num_channels]`` -- so a non-causal
    checkpoint loads into this module unchanged.
    """

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: [B, C, T, Q] ; normalize over (C, Q) for each (B, T) frame.
        assert x.dim() == 4, f"expected 4D [B,C,T,Q], got {x.dim()}D"
        mu = x.mean(dim=(1, 3), keepdim=True)                      # [B,1,T,1]
        var = x.var(dim=(1, 3), unbiased=False, keepdim=True)      # [B,1,T,1]
        x = (x - mu) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class Tar_Model(nn.Module):

    def __init__(
        self,
        stft,
        istft,
        real_att,
        n_freqs,
        hidden_channels,
        n_head,
        emb_dim,
        emb_ks,
        emb_hs,
        num_layers=6,
        eps = 1e-5,
        causal=False,
        p_rnn=None,
        p_attn=None,
        p_conv=None,
        p_norm=None,
        p_inorm=None,
        inorm_warmup=2048,
        inter_block=None,
        grad_checkpoint=False,
    ):
        super(Tar_Model, self).__init__()
        self.num_layers = num_layers
        # OPT-IN activation/gradient checkpointing on the GridNet block stack.
        # Default False -> identical behaviour/memory for every existing config
        # (the running runs are byte-for-byte unaffected). When True, each
        # dual_mdl[i] block is wrapped in torch.utils.checkpoint: forward
        # activations are NOT stored and are recomputed in the backward pass,
        # trading ~1.3x compute for a large activation-memory reduction. Used by
        # the TRUE length-match (~17s chunk) FT so the long-sequence activations
        # fit a single 24GB card. use_reentrant=False (modern, grad-safe).
        self.grad_checkpoint = bool(grad_checkpoint)
        # BLOCK-ONLINE: if set (int W), each GridNetV2Block's time-axis
        # inter_rnn is built bidirectional and run over non-overlapping W-frame
        # chunks -> bounded (W-1)-frame look-ahead. Takes precedence over p_rnn
        # for the inter_rnn structure (so full bidirectional weights load
        # verbatim). All other causal patches (p_attn/p_conv/p_norm/p_inorm)
        # still apply as configured.
        self.inter_block = inter_block

        # ---- granular causal toggles -------------------------------------- #
        # `causal` is the master switch; each individual patch can still be
        # overridden so we can validate them in stages (PC23 strategy):
        #   p_rnn   : patch1  inter_rnn bi->uni  (time-axis LSTM)
        #   p_attn  : patch2  time self-attention causal mask
        #   p_conv  : patch3  input/output Conv time center-pad -> causal L-pad
        #   p_norm  : patch4  conv GroupNorm(1,C) time-global -> per-frame
        #   p_inorm : patch5  GLOBAL input-std normalization (forward L140) is
        #             future-dependent: std over the WHOLE signal rescales frame0
        #             by a future scalar -> dominates latency (measured: response
        #             @0ms even with patches1-4). Replace with a causal warmup-
        #             window std (constant scalar from the first `inorm_warmup`
        #             samples only). NOTE: not in the original 4-patch spec but
        #             empirically REQUIRED for Track-1 causality.
        self.p_rnn = causal if p_rnn is None else p_rnn
        self.p_attn = causal if p_attn is None else p_attn
        self.p_conv = causal if p_conv is None else p_conv
        self.p_norm = causal if p_norm is None else p_norm
        self.p_inorm = causal if p_inorm is None else p_inorm
        self.inorm_warmup = inorm_warmup
        # `self.causal` retained for back-compat / introspection.
        self.causal = bool(
            self.p_rnn or self.p_attn or self.p_conv or self.p_norm or self.p_inorm
        )

        self.stft = stft
        self.istft = istft

        self.att = real_att

        t_ksize = 3
        self.t_ksize = t_ksize
        # patch3: drop the time-axis (center) padding from the conv and instead
        # LEFT-pad the time axis in forward(), so the conv never sees future
        # frames. Frequency-axis padding (=1) is always untouched.
        if self.p_conv:
            ks, padding = (t_ksize, 3), (0, 1)
        else:
            ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        # patch4: GroupNorm(1, C) over [B,C,T,Q] mixes statistics across the
        # whole time axis (uses future frames). Replace with per-frame
        # normalization (each time frame normalized independently over C,Q)
        # keeping the SAME affine weight/bias shape [emb_dim] so the checkpoint
        # loads unchanged.
        if self.p_norm:
            norm = _CausalPerFrameGroupNorm(emb_dim, eps=eps)
        else:
            norm = nn.GroupNorm(1, emb_dim, eps=eps)

        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            norm,
        )
        self.deconv = nn.ConvTranspose2d(2*emb_dim, 2, ks, padding=padding)


        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    GridNetV2Block(
                        2*emb_dim,
                        emb_ks,
                        emb_hs,
                        n_freqs,
                        hidden_channels,
                        n_head,
                        approx_qk_dim=512,
                        activation="prelu",
                        p_rnn=self.p_rnn,
                        p_attn=self.p_attn,
                        inter_block=self.inter_block,
                    )
                )
            )



    def forward(self, input, aux):

        # [B, N, L]
        input = input.unsqueeze(1)
        aux  = aux.unsqueeze(1)

        if self.p_inorm:
            # patch5 (causal input-norm): compute the scaling std from only the
            # leading `inorm_warmup` samples -> a single FUTURE-INDEPENDENT
            # scalar. Same scalar multiplies the output (est * std) so the
            # input/output gain stays consistent. For signals shorter than the
            # warmup window this degrades gracefully to the available prefix.
            w = min(self.inorm_warmup, input.shape[-1])
            std = input[..., :w].std(dim=(1, 2), keepdim=True)
        else:
            std = input.std(dim=(1, 2), keepdim=True)
        input = input / std

        mix_c = self.stft(input)[-1]
        aux_c = self.stft(aux / aux.std(dim=(1, 2), keepdim=True))[-1]

        mix_ri = torch.cat([mix_c.real, mix_c.imag],dim = 1)
        mix_ri = mix_ri.permute(0,1,3,2).contiguous()

        aux_ri = torch.cat([aux_c.real, aux_c.imag],dim = 1)
        aux_ri = aux_ri.permute(0,1,3,2).contiguous()

        if self.p_conv:
            # tensors are [B, 2, T, F]; time axis = dim 2. Left-pad time by
            # (t_ksize-1) so the (time-pad=0) Conv2d sees only past+current
            # frames. F.pad order is (F_left, F_right, T_left, T_right).
            mix_ri = self.conv(F.pad(mix_ri, (0, 0, self.t_ksize - 1, 0)))
            aux_ri = self.conv(F.pad(aux_ri, (0, 0, self.t_ksize - 1, 0)))
        else:
            mix_ri = self.conv(mix_ri)
            aux_ri = self.conv(aux_ri)

        aux_ri = self.att(mix_ri, aux_ri)

        x = torch.cat([mix_ri,aux_ri], dim=1)


        for i in range(self.num_layers):

            if self.grad_checkpoint and self.training:
                # recompute this block's activations in backward (saves VRAM).
                # use_reentrant=False is the modern, autograd-correct variant.
                x = torch.utils.checkpoint.checkpoint(
                    self.dual_mdl[i], x, use_reentrant=False)
            else:
                x = self.dual_mdl[i](x)

        if self.p_conv:
            # deconv has time-pad=0 -> output time = T_in + (t_ksize-1).
            # Trim the trailing (future-side) frames to preserve length and
            # keep the mapping causal (frame t never depends on frames > t).
            T_in = x.shape[2]
            x = self.deconv(x)
            x = x[:, :, :T_in, :].contiguous()
        else:
            x = self.deconv(x)

        out_r = x[:,0,:,:].permute(0,2,1).contiguous()
        out_i = x[:,1,:,:].permute(0,2,1).contiguous()

        est_source = self.istft((out_r, out_i), input_type="real_imag").unsqueeze(1)

        est_source = est_source * std

        return est_source.squeeze(1)