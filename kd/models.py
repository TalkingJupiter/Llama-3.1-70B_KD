from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

TARGET_MODELS_DEFAULT = ["q_proj", "k_proj", "v_proj", "o_proj"]

def load_student(model_id: str,
                lora_r: int = 16,
                lora_alpha: int = 32,
                lora_dropout: float = 0.05,
                target_modules: Optional[List[str]] = None,
                dtype: torch.dtype = torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_i, torch_dtype=dtype)
    model.gradient_checkpointing_enable()
    tm = target_modules or TARGET_MODELS_DEFAULT
    lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=tm, task_type="CASUAL_LM")
    model = get_peft_model(model, lcfg)
    
    return model, tok
