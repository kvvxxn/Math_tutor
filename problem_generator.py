import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from langchain_community.vectorstores import Chroma
from retriever import create_vectorstore, parsing_user_input, retrieve, embed_model

device = "cuda" if torch.cuda.is_available() else "cpu"

CIRICULUM_FILE = "교육과정총정리.md"

# input 받기
user_input = input("문제 생성을 위해 원하는 학년과 학기의 단원 입력하세요 (예: 초등 3학년 2학기 나눗셈) : ")

# input 파싱
content_file, ciriculum_query, course, grade, semester, unit = parsing_user_input(user_input)

# VectorStore 생성
ciriculum_vectorstore = create_vectorstore(CIRICULUM_FILE, name="ciriculum_vectorstore")
ciriculum_query = embed_model.embed_query(ciriculum_query)
retriever = ciriculum_vectorstore.as_retriever()
retrieved_docs = retrieve(retriever, ciriculum_query, k=3)

# Content vectorstore retrieve 용 Query 생성 및 Embedding
query = "Based on the following curriculum content, provide specific achievement criteria, representative examples, and concepts.\n\n"
query = query + "\n".join([doc.page_content for doc in retrieved_docs])
query = embed_model(query)

content_vectorstore = create_vectorstore(content_file, name="content_vectorstore")
retriever = content_vectorstore.as_retriever()
retrieved_docs = retrieve(retriever, query, k=4)


# ------------------------ Retrieved Docs 출력 ------------------------
print("---- Retrieved Curriculum Docs ----")
for doc in retrieved_docs:
    print(doc.page_content)

# ------------------------ LLM Prompt 정의 ------------------------
system_prompt = (
    "You are an expert Korean curriculum math problem setter." 
    "Based on the provided curriculum content, create a new math problem and its solution "
    "suitable for the specified grade and semester in the South Korean 2022 curriculum."
)

user_prompt = (
    f"Using the following curriculum content for {course} {grade}grade {semester} semester's {unit}, "
    "generate a new math problem and its solution.\n\n"
    "Curriculum Content:\n"
)

for doc in retrieved_docs:
    retrieved_content = doc.page_content
    user_prompt += "{retrieved_content}\n\n"

user_prompt += (
    "Please provide the output in the following format:\n"
    "Problem: <math problem here>\n"
    "Solution: <detailed solution here>"
)

# ------------------------ GPT 응답 생성 ------------------------
# GPT-OSS-20B model Load
gpt_model_id = "openai/gpt-oss-20b"
# GPT-OSS-20B Inference
gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_id, use_fast=True)
gpt_model = AutoModelForCausalLM.from_pretrained(
    gpt_model_id,
    torch_dtype="auto",
    device_map="auto",           
)

gpt_messages = [
    {"role": "system", "content": "You are an expert Korean curriculum math problem setter."},
    {"role": "user", "content": "대한민국의 2022 교육과정의 중학교 2학년 2학기에 맞는 새로운 수학 문제와 해설 쌍을 생성해줘."},
]

gpt_prompt = gpt_tokenizer.apply_chat_template(
    gpt_messages, tokenize=False, add_generation_prompt=True
)

gpt_inputs = gpt_tokenizer([gpt_prompt], return_tensors="pt").to(device)
with torch.inference_mode():
    gpt_out = gpt_model.generate(
        **gpt_inputs,
        max_new_tokens=256,
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
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Qwen2.5-VL-7B Instruct Inference
qwen_messages = [
    {
        "role": "system", 
        "content": "You are an expert Korean curriculum problem setter."
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "대한민국의 2022 교육과정의 중학교 2학년 2학기에 맞는 새로운 수학 문제와 해설 쌍을 생성해줘."},
        ],
    }
]

text = processor.apply_chat_template(
    qwen_messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=[text],
    images=None,
    videos=None,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
with torch.inference_mode():
    generated_ids =  qwen_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Qwen2.5-VL-7B Instruct Output")
print(output_text[0])