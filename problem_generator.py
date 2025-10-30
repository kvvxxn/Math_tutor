import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from langchain_community.vectorstores import Chroma
from retriever import create_vectorstore, parsing_user_input, retrieve, embed_model

device = "cuda" if torch.cuda.is_available() else "cpu"

CURRICULUM_FILE = "교육과정총정리.md"

# input 받기
user_input = input("문제 생성을 위해 원하는 학년과 학기의 단원 입력하세요 (예: 초등 3학년 2학기 나눗셈) : ")

# input 파싱
content_file, curriculum_query, course, grade, semester, unit = parsing_user_input(user_input)

# VectorStore 로드 또는 생성
curriculum_vectorstore = create_vectorstore(
    CURRICULUM_FILE, 
    name="curriculum_vectorstore",
    persist_directory="./vectordb/curriculum"
)
curriculum_query = embed_model.embed_query(curriculum_query)
retriever = curriculum_vectorstore.as_retriever()
retrieved_docs = retrieve(retriever, curriculum_query, k=3)

# Content vectorstore retrieve 용 Query 생성 및 Embedding
query = "Based on the following curriculum content, provide specific achievement criteria, representative examples, and concepts.\n\n"
query = query + "\n".join([doc.page_content for doc in retrieved_docs])
query = embed_model(query)

# Content VectorStore 생성
content_vectorstore = create_vectorstore(
    content_file, 
    name="content_vectorstore",
    persist_directory="./vectordb/content"
)
retriever = content_vectorstore.as_retriever()
retrieved_docs = retrieve(retriever, query, k=4)


# ------------------------ Retrieved Docs 출력 ------------------------
print("---- Retrieved Curriculum Docs ----")
for doc in retrieved_docs:
    print(doc.page_content)

# ------------------------ LLM Prompt 정의 ------------------------
system_prompt = (
    "You are an expert Korean curriculum math problem setter." 
    "Based on the provided curriculum content, create a new math problem"
    "suitable for the specified grade and semester in the South Korean 2022 curriculum."
)

user_prompt = (
    f"Using the following curriculum details for {course} {grade}grade {semester} semester's {unit}, "
    "generate a new math problem based on curriculum detail's achivements.\n\n"
    "Choose the type (Multiple choices / Subjective) and Difficulty (상 / 중 / 하) that best suit the given unit and purpose.\n"
    "Curriculum Details:\n"
)

# Retrieved Docs 추가
for doc in retrieved_docs:
    retrieved_content = doc.page_content
    user_prompt += f"{retrieved_content}\n\n"

# Format Instructions
user_prompt += (
    "Please provide the output in the following format:\n"
    "1. Problem Type: 'multiple choice' or 'subjective'\n"
    "2. Problem: [The math problem statement] (난이도: [하, 중, 상])\n"
    "3. Choices: [If multiple choice, list 5 options separated by commas. If subjective, write 'N/A']\n\n"
)

# Three-shot Examples
user_prompt += (
    "Example Format of multiple choice:\n"
    "Problem: 좌표평면 위의 두 점 (1, -1), (2, 1)을 지나는 직선의 Y절편은? (난이도: 하)\n"
    "Choices: 1. -3, 2. -2, 3. -1, 4. 0, 5. 1\n\n"

    "Example Format of multiple choice:\n"
    "Problem: 두 자연수 a, b에 대하여 다항식 2x^2 + 9x + k가 (2x+a)(x+b)로 인수분해되도록 하는 실수 k의 최솟값은? (난이도: 중)\n"
    "Choices: 1. 1, 2. 4, 3. 7, 4. 10, 5. 13\n\n"


    "Example Format of subjective\n"
    "Problem: p < q인 두 소수 p, q에 대하여 p^2q < n <= pq^2을 만족하는 자연수 n의 개수가 308개일 때, p+q를 구하시오. (난이도: 상)\n\n"
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
        "content": system_prompt
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
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