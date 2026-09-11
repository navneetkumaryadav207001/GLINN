"""Multi-Token MLP Projector and Aligned AutoVerbalizer for Chess NLA."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class MultiTokenProjector(nn.Module):
    """2-Layer MLP Projector expanding a 768-dim latent vector into K virtual soft tokens.
    
    Architecture:
      v ∈ ℝ⁷⁶⁸ → Linear(768, 896) → LayerNorm → GELU → Linear(896, K × 896) → Reshape(K, 896) → LayerNorm
    """
    def __init__(
        self,
        d_in: int = 768,
        d_out: int = 896,
        num_tokens: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_tokens = num_tokens

        self.input_layer = nn.Linear(d_in, d_out)
        self.ln_mid = nn.LayerNorm(d_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.expand_layer = nn.Linear(d_out, num_tokens * d_out)
        self.ln_out = nn.LayerNorm(d_out)

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform initialization for smooth gradient flow
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_uniform_(self.expand_layer.weight)
        nn.init.zeros_(self.expand_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects [B, 768] vector into [B, K, 896] soft token embeddings."""
        B = x.shape[0]
        h = self.input_layer(x)
        h = self.ln_mid(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.expand_layer(h)
        h = h.view(B, self.num_tokens, self.d_out)
        out = self.ln_out(h)
        return out


class AlignedAutoVerbalizer(nn.Module):
    """AutoVerbalizer wrapping a frozen (or LoRA) Qwen LLM with a Multi-Token Projector.
    
    The K=8 projected tokens are prepended BEFORE the user prompt without overwriting tokens.
    """
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        d_in: int = 768,
        d_out: int = 896,
        num_tokens: int = 8,
        freeze_backbone: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_tokens = num_tokens

        print(f"[AutoVerbalizer] Loading backbone: {base_model_name}...")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[AutoVerbalizer] Backbone frozen. Only projector is trainable.")

        self.projector = MultiTokenProjector(
            d_in=d_in,
            d_out=d_out,
            num_tokens=num_tokens,
        ).to(device=self.device, dtype=self.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vectors: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        """Prepends K virtual prefix tokens to input_ids and computes causal LM loss."""
        B = vectors.size(0)
        dev = self.device
        vectors = vectors.to(device=dev, dtype=self.dtype)
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)

        # 1. Project vectors into [B, K, d_out]
        prefix_embeds = self.projector(vectors)  # [B, K, 896]

        # 2. Get text embeddings for the prompt + target tokens
        word_embeds = self.backbone.get_input_embeddings()(input_ids)  # [B, SeqLen, 896]

        # 3. Concatenate prefix tokens + text tokens
        inputs_embeds = torch.cat([prefix_embeds, word_embeds], dim=1)  # [B, K + SeqLen, 896]

        # 4. Extend attention mask for prefix tokens
        prefix_mask = torch.ones((B, self.num_tokens), dtype=attention_mask.dtype, device=dev)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 5. Extend labels with -100 for prefix tokens (no loss on virtual prefix)
        full_labels = None
        if labels is not None:
            labels = labels.to(dev)
            prefix_labels = torch.full((B, self.num_tokens), -100, dtype=labels.dtype, device=dev)
            full_labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )
        return outputs

    def generate_explanation(
        self,
        vectors: torch.Tensor,
        tokenizer: Any,
        prompt_text: str = "<|im_start|>user\nAnalyze the chess position activation vector and explain its key dynamics.<|im_end|>\n<|im_start|>assistant\n",
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> list[str]:
        """Autoregressively generates causal explanations conditioned on K prefix tokens with KV caching."""
        self.eval()
        dev = self.device
        B = vectors.size(0)
        vectors = vectors.to(device=dev, dtype=self.dtype)

        # 1. Project vectors into [B, K, d_out]
        prefix_embeds = self.projector(vectors)  # [B, K, 896]

        # 2. Tokenize prompt
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(dev)
        prompt_ids = prompt_ids.repeat(B, 1)  # [B, PromptLen]

        word_embeds = self.backbone.get_input_embeddings()(prompt_ids)
        cur_embeds = torch.cat([prefix_embeds, word_embeds], dim=1)
        attn_mask = torch.ones((B, cur_embeds.size(1)), dtype=torch.long, device=dev)

        generated_tokens: list[list[int]] = [[] for _ in range(B)]
        finished = [False] * B
        eos_id = tokenizer.eos_token_id or tokenizer.encode("<|im_end|>")[0]

        # Initial prompt prefill with KV caching
        with torch.no_grad():
            out = self.backbone(inputs_embeds=cur_embeds, attention_mask=attn_mask, use_cache=True)
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

        for b_idx in range(B):
            tok = next_token[b_idx].item()
            if tok == eos_id:
                finished[b_idx] = True
            else:
                generated_tokens[b_idx].append(tok)

        # Autoregressive generation using single-token step and KV cache
        for _ in range(max_new_tokens - 1):
            if all(finished):
                break
            attn_mask = torch.cat([attn_mask, torch.ones((B, 1), dtype=torch.long, device=dev)], dim=1)
            next_emb = self.backbone.get_input_embeddings()(next_token)
            with torch.no_grad():
                out = self.backbone(
                    inputs_embeds=next_emb,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

            for b_idx in range(B):
                if finished[b_idx]:
                    continue
                tok = next_token[b_idx].item()
                if tok == eos_id:
                    finished[b_idx] = True
                else:
                    generated_tokens[b_idx].append(tok)

        explanations = [tokenizer.decode(t, skip_special_tokens=True).strip() for t in generated_tokens]
        return explanations

    def save_projector(self, save_path: str | Path):
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.projector.state_dict(), p)
        print(f"[Projector] Saved checkpoint -> {p}")

    def load_projector(self, load_path: str | Path):
        self.projector.load_state_dict(torch.load(load_path, map_location=self.device))
        print(f"[Projector] Loaded checkpoint <- {load_path}")
