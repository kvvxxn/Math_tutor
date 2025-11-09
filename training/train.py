import os
import torch
import random
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass

from datasets import load_dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    Mxfp4Config,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# 루트 디렉토리 설정
ROOT_DIR = Path.cwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Masked LM용 인덱스
IGNORE_INDEX = -100

# 훈련 시 이미지+텍스트, 이미지 only, 텍스트 only 비율
MIX_PROBS = {
    "img_text": 0.6,
    "img_only": 0.1,
    "text_only": 0.3,
}

# Training 데이터셋 로드
train_ds = load_dataset("json", data_files="Dataset/labels/train.jsonl", split="train")
val_ds = load_dataset("json", data_files="Dataset/labels/val.jsonl",   split="train")

# ---------- Model load ----------
# 4-bit로 불러오기 위한 설정
quantization_config = Mxfp4Config(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_type=torch.float16,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    quantization_config=quantization_config,
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

model.config.use_cache = False

# ---------- LoRA 적용 ----------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ---------- Preprocessing ----------
seed = random.Random(42)  # 재현성

def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.exists():
        return pp
    alt = ROOT_DIR / p
    return alt if alt.exists() else pp

def _choose_mode_for_train(has_img: bool, has_text: bool) -> str:
    """훈련 샘플에 대해 모드 선택(6:1:3), 불가능한 조합은 자동 폴백."""
    mode = seed.choices(
        population=["img_text", "img_only", "text_only"],
        weights=[MIX_PROBS["img_text"], MIX_PROBS["img_only"], MIX_PROBS["text_only"]],
        k=1,
    )[0]
    # 폴백 로직
    if mode == "img_only" and not has_img:
        mode = "text_only" if has_text else "img_text"
    if mode == "text_only" and not has_text:
        mode = "img_only" if has_img else "img_text"
    if mode == "img_text":
        if not has_img and has_text:
            mode = "text_only"
        elif has_img and not has_text:
            mode = "img_only"
        elif (not has_img) and (not has_text):
            mode = "text_only"
    return mode

def _build_messages(example: Dict[str, Any], mode: str) -> list:
    """
    JSONL 레코드 -> Qwen3-VL chat messages
      mode in {"img_text","img_only","text_only"}
    """
    convs = example.get("conversations", [])
    messages: List[Dict[str, Any]] = []

    # System prompt
    idx = 0
    if convs and convs[0].get("from") == "system":
        messages.append({"role": "system", "content": [{"type": "text", "text": convs[0]["value"]}]})
        idx = 1

    # User prompt
    user_text = ""
    for m in convs[idx:]:
        if m.get("from") == "user":
            user_text = m.get("value", "")
            break

    # Image load
    imgs = []
    for p in example.get("images", []):
        if isinstance(p, str) and p:
            path = _resolve_path(p)
            if path.exists():
                try:
                    imgs.append(Image.open(path).convert("RGB"))
                except Exception:
                    pass

    has_img = len(imgs) > 0
    has_text = isinstance(user_text, str) and len(user_text.strip()) > 0

    # user content 구성 모드별
    content = []
    if mode in ("img_text", "img_only") and has_img:
        content.extend([{"type": "image", "image": im} for im in imgs])
    if mode in ("img_text", "text_only"):
        content.append({"type": "text", "text": user_text if has_text else ""})
    # 혹시 content가 비면 최소 텍스트 하나 넣기
    if not content:
        content.append({"type": "text", "text": user_text})

    messages.append({"role": "user", "content": content})

    # assistant 답변
    assistant_text = ""
    for m in convs[idx:]:
        if m.get("from") == "assistant":
            assistant_text = m.get("value", "")
            break
    messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

    return messages

def _mask_assistant_only(model_inputs, tokenizer) -> Dict[str, torch.Tensor]:
    """
    processor(messages=...) 출력에서 마지막 assistant 본문만 labels로 두고 나머지는 -100.
    """
    input_ids = model_inputs["input_ids"][0]
    attention_mask = model_inputs["attention_mask"][0]

    tok = tokenizer
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end   = tok.convert_tokens_to_ids("<|im_end|>")
    t_system    = tok.convert_tokens_to_ids("system")
    t_user      = tok.convert_tokens_to_ids("user")
    t_assistant = tok.convert_tokens_to_ids("assistant")

    starts = (input_ids == im_start).nonzero(as_tuple=True)[0].tolist()
    aspan = None
    for s in starts:
        if s + 1 < len(input_ids) and input_ids[s + 1].item() == t_assistant:
            tail = input_ids[s + 2 :]
            rel_end = (tail == im_end).nonzero(as_tuple=True)
            end = (s + 2 + rel_end[0][0].item()) if len(rel_end[0]) > 0 else len(input_ids)
            aspan = (s + 2, end)

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    if aspan is not None:
        a, b = aspan
        special = {im_start, im_end, tok.eos_token_id, tok.bos_token_id, t_system, t_user, t_assistant}
        while a < b and input_ids[a].item() in special:
            a += 1
        if a < b:
            labels[a:b] = input_ids[a:b]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    for k, v in model_inputs.items():
        if k not in out:
            out[k] = v[0]
    return out

def preprocess_train(example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # 훈련 전용: 6:1:3 혼합
    # 가능한 리소스를 파악해서 모드 선택
    has_img = any(Path(p).exists() or (ROOT_DIR / p).exists() for p in example.get("images", []))
    has_text = False
    convs = example.get("conversations", [])
    start_idx = 1 if convs and convs[0].get("from") == "system" else 0
    for m in convs[start_idx:]:
        if m.get("from") == "user" and isinstance(m.get("value"), str) and m["value"].strip():
            has_text = True
            break
    mode = _choose_mode_for_train(has_img, has_text)

    messages = _build_messages(example, mode)
    model_inputs = processor(messages=messages, return_tensors="pt")
    return _mask_assistant_only(model_inputs, processor.tokenizer)

def preprocess_val(example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # 검증: 데이터에 있는 그대로(이미지도, 텍스트도 모두 사용)
    messages = _build_messages(example, mode="img_text")
    model_inputs = processor(messages=messages, return_tensors="pt")
    return _mask_assistant_only(model_inputs, processor.tokenizer)

# ---------- map 전처리 ----------
train_pp = train_ds.map(preprocess_train, remove_columns=train_ds.column_names)
val_pp   = val_ds.map(preprocess_val,   remove_columns=val_ds.column_names)

# ---------- Collator ----------
@dataclass
class CollatorQwenVL:
    pad_id: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ids  = [f["input_ids"] for f in features]
        attn = [f["attention_mask"] for f in features]
        labs = [f["labels"] for f in features]
        ids  = torch.nn.utils.rnn.pad_sequence(ids,  batch_first=True, padding_value=self.pad_id)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
        labs = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=IGNORE_INDEX)
        batch = {"input_ids": ids, "attention_mask": attn, "labels": labs}

        # 멀티모달 키(있을 때만): pixel_values / image_grid_thw 등
        for key in ["pixel_values", "image_grid_thw"]:
            if key in features[0]:
                vals = [f.get(key) for f in features if key in f]
                if len(vals) > 0 and isinstance(vals[0], torch.Tensor):
                    batch[key] = torch.stack(vals)
                elif len(vals) > 0:
                    batch[key] = vals
        return batch

collator = CollatorQwenVL(pad_id=processor.tokenizer.pad_token_id or 0)

# ---------- TrainingArguments / Trainer ----------
args = TrainingArguments(
    output_dir="./qwen3vl_math_lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_pp,
    eval_dataset=val_pp,
    data_collator=collator,
)

trainer.train()