import os
from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.metrics import kl_div_loss, resolve_loss_fn
from speculators.models.utils import resolve_target_layer_ids

if os.environ.get("DFLASH_FUSED_RMSNORM") == "1":
    # Replace every Qwen3RMSNorm (model norms + per-layer norms + attention
    # q/k norms, ~23 instances) with the fused NPU RMSNorm kernel: one kernel
    # vs the eager cast->pow->mean->rsqrt->mul->cast chain (~3.9x/op, fewer
    # launches + kills the fp32 up/down Casts). bf16-approximate, not bit-exact
    # -> convergence validated by A/B (OPEN_ISSUES #2 / EXPERIMENTS).
    import torch_npu  # noqa: PLC0415

    def _npu_fused_rmsnorm_forward(self, hidden_states):
        return torch_npu.npu_rms_norm(
            hidden_states, self.weight, epsilon=self.variance_epsilon
        )[0]

    Qwen3RMSNorm.forward = _npu_fused_rmsnorm_forward
    print("[dflash] fused NPU RMSNorm ON (DFLASH_FUSED_RMSNORM=1)", flush=True)


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(
        self,
        config: DFlashSpeculatorConfig,
    ) -> None:
        # Forcibly override config settings
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        self._attn_impl = config.transformer_layer_config._attn_implementation  # noqa: SLF001
        self._create_mask_fn = (
            create_block_mask
            if self._attn_impl == "simple_flex_attention"
            else create_mask
        )
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config

        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx)  # type: ignore[arg-type]
                for layer_idx in range(num_draft_layers)
            ]
        )
        self.sliding_window = tl_config.sliding_window
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(tl_config.layer_types)
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_draft_layers - len(self.sliding_window_indices))
        self.sliding_window_non_causal = config.sliding_window_non_causal

        self.norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config.transformer_layer_config)  # type: ignore[arg-type]

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.transformer_layer_config.hidden_size,
            config.transformer_layer_config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False
        self.block_size = config.block_size
        # Optional NPU mask-kernel fast-path for the dense full-attention mask
        # (OPEN_ISSUES #2). Off by default and gated by env, so baseline vs
        # kernel runs are a single flag flip. Only replaces create_mask on the
        # full-attention (sliding_window=None) sdpa/eager path.
        self._use_mask_kernel = os.environ.get("DFLASH_USE_MASK_KERNEL", "0") == "1"
        self.post_init()

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlashDraftModel":
        """Create DFlash model from training arguments.

        Args:
            verifier_config: Verifier model configuration. This should be a config
                with num_hidden_layers set to the number of DRAFT layers (created
                by create_transformer_layer_config in train.py).
            t2d: Target-to-draft vocabulary mapping tensor (optional)
            d2t: Draft-to-target vocabulary mapping tensor (optional)
            **kwargs: Training arguments with DFlash-specific params
                - draft_vocab_size: Size of draft vocabulary
                - block_size: Block size for draft predictions (default: 8)
                - max_anchors: Max anchor positions during training (default: 256)
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized DFlashDraftModel

        Note:
            The number of draft layers is encoded in verifier_config.num_hidden_layers,
            following the same pattern as EAGLE3.
        """
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        # Resolve target layer IDs if not provided
        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )

        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )

        config = DFlashSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=kwargs.get("block_size", 8),
            max_anchors=kwargs.get("max_anchors", 3072),
            aux_hidden_state_layer_ids=target_layer_ids,
            mask_token_id=kwargs.get("mask_token_id"),
            sliding_window_non_causal=kwargs.get("sliding_window_non_causal", False),
            speculators_config=SpeculatorsConfig(
                algorithm="dflash",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        # DFlash first position is anchor position, not used during gen
                        speculative_tokens=kwargs.get("block_size", 8) - 1,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for DFlash.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_fn = resolve_loss_fn(kwargs["loss_fn"])
        return {"loss_fn": loss_fn}, {"loss_fn": loss_fn}

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    def _create_attention_mask(
        self,
        lengths: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
        sliding_window_non_causal: bool = False,
    ):
        # NPU mask-kernel fast-path: bit-exact drop-in for create_mask on the
        # full-attention path (sliding_window=None). Falls back to create_mask for
        # the sliding-window and flex (create_block_mask) paths, which the kernel
        # does not cover. dflash_mask builds metadata on the input device, so pass
        # the on-device tensors directly (no D2H sync on the training critical path).
        # See OPEN_ISSUES #2 / TRAINING_INTEGRATION.md.
        if (
            self._use_mask_kernel
            and sliding_window is None
            and self._create_mask_fn is create_mask
        ):
            from dflash_mask import mask as _kernel_mask  # noqa: PLC0415

            if not getattr(self, "_mask_kernel_logged", False):
                print(
                    "[dflash] mask via NPU kernel (DFLASH_USE_MASK_KERNEL=1)",
                    flush=True,
                )
                self._mask_kernel_logged = True
            return _kernel_mask(
                lengths,
                total_seq_len,
                anchor_positions,
                self.block_size,
                device,
            )
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            lengths=lengths.to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            sliding_window=sliding_window,
            sliding_window_non_causal=sliding_window_non_causal,
        )
        return self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    @torch.compiler.disable
    def _build_attention_mask(
        self,
        loss_mask,
        lengths,
        device,
        *,
        anchor_positions=None,
        anchor_valid=None,
        mask_metadata=None,
    ):
        total_seq_len = loss_mask.shape[1]

        # anchors may be precomputed in the dataloader (OPEN_ISSUES #2); else select here.
        if anchor_positions is None:
            anchor_positions, anchor_valid = select_anchors(
                loss_mask, self.config.max_anchors, self.block_size
            )

        full_attn_mask = None
        if self.uses_full_attn:
            if mask_metadata is not None and self._use_mask_kernel:
                # precomputed kernel metadata -> call the kernel directly, keeping
                # select_anchors / build_metadata off the training critical path.
                from dflash_mask import mask_from_metadata  # noqa: PLC0415

                if not getattr(self, "_mask_kernel_meta_logged", False):
                    print(
                        "[dflash] mask via NPU kernel (dataloader-precomputed metadata)",
                        flush=True,
                    )
                    self._mask_kernel_meta_logged = True
                full_attn_mask = mask_from_metadata(
                    *mask_metadata, self.block_size, device
                )
            else:
                full_attn_mask = self._create_attention_mask(
                    lengths=lengths,
                    total_seq_len=total_seq_len,
                    anchor_positions=anchor_positions,
                    device=device,
                    sliding_window=None,
                )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._create_attention_mask(
                lengths=lengths,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal,
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_fn=kl_div_loss,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = self.config.max_anchors

        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        # Optional dataloader-precomputed anchors + mask-kernel metadata (popped
        # from kwargs so they don't flow into the decoder layers). OPEN_ISSUES #2.
        pre_anchor_positions = kwargs.pop("anchor_positions", None)
        pre_anchor_valid = kwargs.pop("anchor_valid", None)
        _kv_doc = kwargs.pop("mask_kv_doc", None)
        _kv_iota = kwargs.pop("mask_kv_iota", None)
        _q_doc = kwargs.pop("mask_q_doc", None)
        _q_anchor = kwargs.pop("mask_q_anchor", None)
        _blk_idx = kwargs.pop("mask_blk_idx", None)
        pre_mask_metadata = (
            (_kv_doc, _kv_iota, _q_doc, _q_anchor, _blk_idx)
            if _kv_doc is not None
            else None
        )

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(
                loss_mask,
                lengths,
                device,
                anchor_positions=pre_anchor_positions,
                anchor_valid=pre_anchor_valid,
                mask_metadata=pre_mask_metadata,
            )
        )

        mask_tokens_size = num_anchors * self.block_size

        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )  # shape: [1, num_anchors*block_size]
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)
        # shape: [1, num_anchors*block_size, hidden_size]

        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)
        # shape: [1, total_seq_len, hidden_size]

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[:, anchor_positions], self.block_size, input_ids.numel()
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size]

        # the hidden_states shape doesn't match position_ids but doesn't need
        # to, as hidden_states is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size, input_ids.numel()
        )  # shape: [num_anchors*block_size]

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            # Shift right by 1 so verifier_logits[i] predicts token at position i
            verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]
            # shape: [1, num_anchors*block_size, draft_vocab_size]

        for layer_idx, layer in enumerate(self.layers):
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        logits = self.lm_head(self.norm(noise_embedding))
        # shape: [1, num_anchors*block_size, vocab_size]

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size]

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, num_anchors*block_size]

        aligned_loss_mask[:, :: self.block_size] = 0
        loss, metrics = compute_metrics(
            logits, targets, aligned_loss_mask, self.block_size, loss_fn=loss_fn
        )
        draft_tokens = torch.argmax(logits, dim=-1)

        return draft_tokens, loss, metrics
