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
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from math_tutor.config.path import DATA_DIR, ORIGINAL_DATA_DIR, ROOT_DIR

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

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
train_ds = load_dataset("json", data_files=(DATA_DIR / "Prepared" / "train.jsonl").as_posix(), split="train")
val_ds = load_dataset("json", data_files=(DATA_DIR / "Prepared" / "val.jsonl").as_posix(), split="train")

# ---------- Model load ----------
# 4-bit로 불러오기 위한 설정
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    quantization_config=quantization_config,
    dtype="auto", 
    device_map="auto"
)

# Processor 로드
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# required_grad 설정 및 gradient checkpointing 활성화
model = prepare_model_for_kbit_training(
    model, 
    use_gradient_checkpointing=True
)

# Gradient Checkpointing 활성화 시 꺼야 하는 설정
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
seed = random.Random(42)  

def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.exists():
        return pp
    alt = ROOT_DIR / p
    return alt if alt.exists() else pp

def _choose_mode_for_train(has_img: bool, has_text: bool) -> str:
    """
    6:3:1의 비율로 Imaeg+text, image only, text only 중 하나 선택
    """
    mode = seed.choices(
        population=["img_text", "img_only", "text_only"],
        weights=[MIX_PROBS["img_text"], MIX_PROBS["img_only"], MIX_PROBS["text_only"]],
        k=1,
    )[0]
    
    # 가능하지 않은 경우 처리
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

def _text_seg(x: Any) -> Dict[str, str]:
    """
    항상 text segment로 강제
    text segment: {"type":"text","text":...}
    QWEN3-VL은 text segment만 지원하기 때문
    """
    
    if x is None:
        x = ""
    if not isinstance(x, str):
        x = str(x)
    return {"type": "text", "text": x}

def _build_messages(example: Dict[str, Any], mode: str) -> list:
    """
    JSONL 레코드 -> Qwen3-VL chat messages
    모든 role의 content를 text segment list로 통일함.
    """
    convs = example.get("conversations", []) or []
    messages: List[Dict[str, Any]] = []

    # ----- system prompt -----
    idx = 0
    if convs and convs[0].get("from") == "system":
        sys_txt = convs[0].get("value", "")
        messages.append({"role": "system", "content": [_text_seg(sys_txt)]})
        idx = 1

    # ----- user prompt -----
    user_text = ""
    for m in convs[idx:]:
        if (m.get("from") == "user"):
            user_text = m.get("value", "")
            break
    user_text = "" if user_text is None else (user_text if isinstance(user_text, str) else str(user_text))

    # ----- 이미지 로드 -----
    imgs: List[Image.Image] = []
    for p in example.get("images", []) or []:
        if isinstance(p, str) and p:
            path = _resolve_path(p)
            if path.exists():
                try:
                    imgs.append(Image.open(path).convert("RGB"))
                except Exception:
                    pass

    has_img = len(imgs) > 0
    has_text = len(user_text.strip()) > 0

    # ----- user content -----
    content: List[Dict[str, Any]] = []
    if mode in ("img_text", "img_only") and has_img:
        for im in imgs:
            content.append({"type": "image", "image": im})
    if mode in ("img_text", "text_only"):
        content.append(_text_seg(user_text if has_text else ""))

    if not content:
        content.append(_text_seg(user_text))

    messages.append({"role": "user", "content": content})

    # ----- assistant -----
    assistant_text = ""
    for m in convs[idx:]:
        if (m.get("from") == "assistant"):
            assistant_text = m.get("value", "")
            break
    assistant_text = "" if assistant_text is None else (assistant_text if isinstance(assistant_text, str) else str(assistant_text))

    messages.append({"role": "assistant", "content": [_text_seg(assistant_text)]})

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

    out = {k: v[0] for k, v in model_inputs.items()}
    
    out["input_ids"] = input_ids
    out["attention_mask"] = attention_mask
    out["labels"] = labels

    # Key error 방지: 'pixel_values', 'image_grid_thw'가 없을 경우 None 할당
    if "pixel_values" not in out:
        out["pixel_values"] = None
    if "image_grid_thw" not in out:
        out["image_grid_thw"] = None

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

    # 이미지 추출
    images = []
    for msg in messages:
        if msg["role"] == "user":
            for segment in msg["content"]:
                if segment["type"] == "image":
                    images.append(segment["image"])
    
    # messages 리스트에 QWEN3-VL Chat template 적용
    # Tokenization은 Processor 내부에서 수행
    # Assistant 부분이 이미 존재하기 때문에 add_generation_prompt=False
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # processor 호출
    model_inputs = processor(
        text=text,
        images=images if images else None,  # 이미지가 있을 때만 전달
        return_tensors="pt"
    )

    # 정답 부분 Masking
    return _mask_assistant_only(model_inputs, processor.tokenizer)

def preprocess_val(example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    messages = _build_messages(example, mode="img_text")

    # 이미지 추출
    images = []
    for msg in messages:
        if msg["role"] == "user":
            for segment in msg["content"]:
                if segment["type"] == "image":
                    images.append(segment["image"])

    # messages 리스트에 QWEN3-VL Chat template 적용
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # processor 호출
    model_inputs = processor(
        text=text,
        images=images if images else None,
        return_tensors="pt"
    )
    
    # 정답 부분 Masking
    return _mask_assistant_only(model_inputs, processor.tokenizer)

# ---------- JSONL -> Dataset ----------
train_pp = train_ds.map(preprocess_train, remove_columns=train_ds.column_names)
val_pp   = val_ds.map(preprocess_val,   remove_columns=val_ds.column_names)

# ---------- Collator function ----------
@dataclass
class CollatorQwenVL:
    pad_id: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        # Padding
        ids  = [torch.tensor(f["input_ids"]) for f in features]
        attn = [torch.tensor(f["attention_mask"]) for f in features]
        labs = [torch.tensor(f["labels"]) for f in features]
        
        ids  = torch.nn.utils.rnn.pad_sequence(ids,  batch_first=True, padding_value=self.pad_id)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
        labs = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=IGNORE_INDEX)
        batch = {"input_ids": ids, "attention_mask": attn, "labels": labs}

        # 텍스트 있으면 추가
        pixel_values_list = [f.get("pixel_values") for f in features if f.get("pixel_values") is not None]
        if pixel_values_list:
            batch["pixel_values"] = torch.stack([torch.tensor(pv) for pv in pixel_values_list])

        # 이미지 있으면 추가
        image_grid_thw_list = [f.get("image_grid_thw") for f in features if f.get("image_grid_thw") is not None]
        if image_grid_thw_list:
            batch["image_grid_thw"] = torch.tensor(image_grid_thw_list)

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
    eval_strategy="steps",
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