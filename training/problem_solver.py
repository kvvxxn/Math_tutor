import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training 데이터셋 로드

# Validation 데이터셋 로드

# 4-bit로 불러오기 위한 설정
quantization_config = Mxfp4Config(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_type=torch.float16,
)

# Model 로드
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", 
    quantization_config=quantization_config,
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# LoRA 적용
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Trainer 설정
sft_config = SFTConfig(
    train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    output_dir="./qwen3vl_lora_finetuned",
)
trainer = SFTTrainer(
    model=model,
    train_dataset=None,  
    peft_config=lora_config,
    sft_config=sft_config,
    tokenizer=processor.tokenizer,
)

# 모델 학습 시작
trainer.train()