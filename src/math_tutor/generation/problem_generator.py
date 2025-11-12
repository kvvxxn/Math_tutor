import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from math_tutor.rag.utils import parsing_user_input, retrieve

device = "cuda" if torch.cuda.is_available() else "cpu"

# input 받기
user_input = input("문제 생성을 위해 원하는 학년과 학기의 단원 입력하세요 (예: 초등 3학년 2학기 나눗셈) : ")

# input 파싱
vectordb, course, grade, semester, unit = parsing_user_input(user_input)

# Content vectorstore retrieve 용 Query 생성 및 Embedding
query = f"'{unit}'에 대한 중요한 개념과 성취 기준."

# Document retrieve
retriever = vectordb.as_retriever()
retrieved_docs = retrieve(retriever, query, k=2)


# ------------------------ Retrieved Docs 출력 ------------------------
print("---- Retrieved Curriculum Docs ----")
for doc in retrieved_docs:
    print(doc.page_content)

# ------------------------ LLM Prompt 정의 ------------------------
system_prompt = (
    "You are an expert Korean curriculum math problem setter, with 20 years of experience teaching math.\n\n"

    "Your primary goal is to create a single, new math problem that a student will actually solve in a classroom.\n\n"

    "CRITICAL RULES:\n"
    "\t1. You MUST generate the problem itself, NOT instructions about how to create a problem.\n"
    "\t2. You MUST NOT generate meta-questions or meta-problems (e.g., \"Create a division problem...\").\n"
    "\t3. Base the problem *only* on the provided curriculum content.\n"
    "\t4. Output MUST be ONLY a single JSON object that follows the exact schema below. Do NOT add any explanations, analysis, \
        reasoning, thoughts, disclaimers, prefaces, or extra text before or after the JSON. Do NOT wrap the JSON in backticks.\n"
    "\t5. For problems involving shapes, only one shape may be included. "
        "This shape MUST be represented as SVG (Scalable Vector Graphics) code placed in the '이미지' field. "
        "If no image is needed, you MUST write 'N/A' in the 'Image' field.\n"
    "\t6. NEVER reveal your chain-of-thought, rationale, or intermediate steps. Think privately, but output ONLY the final JSON object.\n"
)

user_prompt = (
    f"{course} {grade}학년 {semester}학기의 {unit} 단원에 대한 아래 커리큘럼 내용을 이용해서, "
    "커리큘럼의 성취 기준에 맞추어서 하나의 수학 문제를 생성해줘.\n\n"
    "생성할 문제를 위해 주어진 목적과 단원을 바탕으로 두 가지 문제 유형(객관식 / 주관식) 중 하나를 선택해야 하고, 세 가지 난이도(상 / 중 / 하) 중 하나를 선택해야 해.\n"
    "커리큘럼:\n"
)

# Retrieved Docs 추가
for doc in retrieved_docs:
    retrieved_content = doc.page_content
    user_prompt += f"{retrieved_content}\n\n"

# Format Instructions
user_prompt += (
    "한국어로 생성한 문제를 아래 JSON 형식으로 작성하여 제공해줘.\n"
    "{\n"
    '    "problems": [\n'
    '        {"유형": "객관식 / 주관식"},\n'
    '        {"문제": "[The math problem statement]"},\n'
    '        {"난이도": "상 / 중 / 하"},\n'
    '        {"보기": "If multiple choice, list 5 options separated by commas. If subjective, write \'N/A\'"},\n'
    '        {"이미지": "If an image is necessary for the problem, only one shape may be included. If not needed, write \'N/A\'"}\n'
    '    ]\n'
    '}\n\n'
)

# Three-shot Examples
user_prompt += (
    "수학 문제 예시\n\n"
    "{\n"
    '  "problems": [\n'
    '    {\n'
    '      "유형": "객관식",\n'
    '      "문제": "좌표평면 위의 두 점 (1, -1), (2, 1)을 지나는 직선의 Y절편은?",\n'
    '      "난이도": "하",\n'
    '      "보기": "1. -3, 2. -2, 3. -1, 4. 0, 5. 1",\n'
    '      "이미지": "N/A"\n'
    '    }\n'
    '  ]\n'
    '}\n\n'
    "{\n"
    '  "problems": [\n'
    '    {\n'
    '      "유형": "객관식",\n'
    '      "문제": "두 자연수 a, b에 대하여 다항식 2x^2 + 9x + k가 (2x+a)(x+b)로 인수분해되도록 하는 실수 k의 최솟값은?",\n'
    '      "난이도": "중",\n'
    '      "보기": "1. 1, 2. 4, 3. 7, 4. 10, 5. 13",\n'
    '      "이미지": "N/A"\n'
    '    }\n'
    '  ]\n'
    '}\n\n'
    "{\n"
    '  "problems": [\n'
    '    {\n'
    '      "유형": "주관식",\n'
    '      "문제": "p < q인 두 소수 p, q에 대하여 p^2q < n <= pq^2을 만족하는 자연수 n의 개수가 308개일 때, p+q를 구하시오.",\n'
    '      "난이도": "상",\n'
    '      "보기": "N/A",\n'
    '      "이미지": "N/A"\n'
    '    }\n'
    '  ]\n'
    '}\n\n'
)


# ------------------------ GPT 응답 생성 ------------------------
# GPT-OSS-20B model Load
gpt_model_id = "openai/gpt-oss-20b"
# GPT-OSS-20B Inference
gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_id, use_fast=True)
gpt_model = AutoModelForCausalLM.from_pretrained(
    gpt_model_id,
    dtype="auto",
    device_map="auto",           
).to(device)

gpt_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

gpt_prompt = gpt_tokenizer.apply_chat_template(
    gpt_messages, tokenize=False, add_generation_prompt=True
)

gpt_inputs = gpt_tokenizer([gpt_prompt], return_tensors="pt").to(device)
with torch.inference_mode():
    gpt_out = gpt_model.generate(
        **gpt_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=gpt_tokenizer.eos_token_id,
        eos_token_id=gpt_tokenizer.eos_token_id,
    )

print("GPT-OSS-20B Output")
print(gpt_tokenizer.decode(gpt_out[0], skip_special_tokens=True))



# ------------------------ QWEN 응답 생성 ------------------------
# Load Qwen2.5-VL-7B Instruct model 
qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Qwen2.5-VL-7B Instruct Inference
qwen_messages = [
    {
        "role": "system", 
        "content": [  
            {"type": "text", "text": system_prompt}
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text", "text": user_prompt
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    qwen_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(qwen_model.device)

# Inference
with torch.inference_mode():
    generated_ids =  qwen_model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True, # Next word 선택 시 Sampling
        temperature=0.7,
        top_p=0.9,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Qwen3-VL-8B Instruct Output")
print(output_text[0])