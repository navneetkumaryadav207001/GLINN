"""Critic (AutoReader) and Verbalizer (AutoVerbalizer) Architectures for Chess NLA."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import NLAConfig
from .injection import extract_explanation, normalize_activation


class CrossDimensionAdapter(nn.Module):
    """Cross-dimension linear adapter bridge with Truncated Identity Initialization."""
    def __init__(self, d_in: int, d_out: int, init_mode: str = "truncated_identity", bias: bool = False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.proj = nn.Linear(d_in, d_out, bias=bias)

        if init_mode == "truncated_identity":
            k = min(d_out, d_in)
            nn.init.zeros_(self.proj.weight)
            self.proj.weight.data[:k, :k] = torch.eye(k)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
        else:
            nn.init.xavier_uniform_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class NLACriticModel(nn.Module):
    """AutoReader (AR): Reads natural language explanations and inverts them back to 768-dim chess vectors."""
    def __init__(self, nla_config: NLAConfig, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        super().__init__()
        self.config = nla_config
        self.device = torch.device(device)
        self.dtype = dtype

        self.backbone = AutoModelForCausalLM.from_pretrained(
            nla_config.base_model,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=nla_config.lora_r,
            lora_alpha=nla_config.lora_alpha,
            lora_dropout=nla_config.lora_dropout,
            target_modules=nla_config.lora_target_modules,
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        # Output projection from Qwen (896) to Chess vector space (768)
        self.proj = CrossDimensionAdapter(
            d_in=nla_config.d_model,
            d_out=nla_config.target_d_model,
            init_mode=nla_config.adapter_init_mode,
            bias=False,
        ).to(device=self.device, dtype=self.dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lengths]
        recon = self.proj(pooled)
        return recon

    def reconstruct_text(
        self,
        explanations: list[str],
        tokenizer: Any,
        prompt_template: str,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Helper to reconstruct activation vectors directly from explanation strings."""
        prompts = [prompt_template.format(explanation=e) for e in explanations]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            pred = self.forward(enc["input_ids"], enc["attention_mask"])
        return pred

    def save_pretrained(self, save_dir: str | Path):
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(p)
        torch.save(self.proj.state_dict(), p / "proj_head.pt")

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str | Path,
        nla_config: NLAConfig,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> NLACriticModel:
        model = cls(nla_config, dtype=dtype, device=device)
        p = Path(save_dir)
        model.backbone = PeftModel.from_pretrained(model.backbone.base_model.model, str(p))
        if (p / "proj_head.pt").exists():
            model.proj.load_state_dict(torch.load(p / "proj_head.pt", map_location=device))
        return model


class NLAVerbalizerModel(nn.Module):
    """AutoVerbalizer (AV): Maps 768-dim chess activation vectors to natural language explanations."""
    def __init__(self, nla_config: NLAConfig, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        super().__init__()
        self.config = nla_config
        self.device = torch.device(device)
        self.dtype = dtype

        self.backbone = AutoModelForCausalLM.from_pretrained(
            nla_config.base_model,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=nla_config.lora_r,
            lora_alpha=nla_config.lora_alpha,
            lora_dropout=nla_config.lora_dropout,
            target_modules=nla_config.lora_target_modules,
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        # Input adapter mapping chess vector space (768) into Qwen residual stream (896)
        self.input_proj = CrossDimensionAdapter(
            d_in=nla_config.target_d_model,
            d_out=nla_config.d_model,
            init_mode=nla_config.adapter_init_mode,
            bias=False,
        ).to(device=self.device, dtype=self.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vectors: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        dev = input_ids.device
        vectors = vectors.to(device=dev, dtype=self.dtype)
        tok_emb = self.backbone.get_input_embeddings()
        word_embs = tok_emb(input_ids)  # [B, SeqLen, 896]

        # Project 768-dim chess vector to 896
        proj_vec = self.input_proj(vectors).unsqueeze(1)  # [B, 1, 896]
        inputs_embeds = torch.cat([proj_vec, word_embs[:, 1:, :]], dim=1)

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        return outputs

    def generate_explanation(
        self,
        vectors: torch.Tensor,
        tokenizer: Any,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> list[str]:
        """Autoregressively generates causal explanations from chess vectors."""
        dev = vectors.device
        B = vectors.size(0)
        proj_vec = self.input_proj(vectors.to(self.dtype)).unsqueeze(1)  # [B, 1, 896]

        prompt_str = "<|im_start|>user\n" + self.config.verbalizer_prompt_template + "<|im_end|>\n<|im_start|>assistant\n"
        enc = tokenizer(prompt_str, return_tensors="pt").to(dev)
        base_ids = enc["input_ids"].repeat(B, 1)  # [B, PromptLen]

        tok_emb = self.backbone.get_input_embeddings()
        word_embs = tok_emb(base_ids)
        cur_embeds = torch.cat([proj_vec, word_embs[:, 1:, :]], dim=1)
        cur_ids = base_ids.clone()
        attn_mask = torch.ones(cur_ids.shape, dtype=torch.long, device=dev)

        generated_tokens: list[list[int]] = [[] for _ in range(B)]
        eos_id = tokenizer.eos_token_id or tokenizer.encode("<|im_end|>")[0]

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.backbone(inputs_embeds=cur_embeds, attention_mask=attn_mask)
                logits = out.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

            cur_ids = torch.cat([cur_ids, next_token], dim=1)
            attn_mask = torch.cat([attn_mask, torch.ones((B, 1), dtype=torch.long, device=dev)], dim=1)
            next_emb = tok_emb(next_token)
            cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)

            all_done = True
            for b_idx in range(B):
                tok = next_token[b_idx].item()
                if tok == eos_id:
                    continue
                generated_tokens[b_idx].append(tok)
                all_done = False
            if all_done:
                break

        explanations = [tokenizer.decode(t, skip_special_tokens=True).strip() for t in generated_tokens]
        return explanations

    def save_pretrained(self, save_dir: str | Path):
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(p)
        torch.save(self.input_proj.state_dict(), p / "input_proj.pt")

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str | Path,
        nla_config: NLAConfig,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> NLAVerbalizerModel:
        model = cls(nla_config, dtype=dtype, device=device)
        p = Path(save_dir)
        model.backbone = PeftModel.from_pretrained(model.backbone.base_model.model, str(p))
        if (p / "input_proj.pt").exists():
            model.input_proj.load_state_dict(torch.load(p / "input_proj.pt", map_location=device))
        return model
