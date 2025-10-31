import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from langchain_community.vectorstores import Chroma
from retriever import create_vectorstore, parsing_user_input, retrieve

device = "cuda" if torch.cuda.is_available() else "cpu"

# input 받기
user_input = input("문제 생성을 위해 원하는 학년과 학기의 단원 입력하세요 (예: 초등 3학년 2학기 나눗셈) : ")

# input 파싱
content_file, curriculum_query, course, grade, semester, unit = parsing_user_input(user_input)

# Content vectorstore retrieve 용 Query 생성 및 Embedding
query = f"Key concepts, achievement criteria, and instructional considerations for the unit '{unit}' of {course}, Grade {grade}, Semester {semester}"

content_vectorstore = create_vectorstore(
    content_file, 
    name="content_vectorstore",
    persist_directory="./vectordb/content"
)
retriever = content_vectorstore.as_retriever()
retrieved_docs = retrieve(retriever, query, k=2)


# ------------------------ Retrieved Docs 출력 ------------------------
print("---- Retrieved Curriculum Docs ----")
for doc in retrieved_docs:
    print(doc.page_content)

# ------------------------ LLM Prompt 정의 ------------------------
system_prompt = (
    "You are an expert Korean curriculum math problem setter, "
    "with 10 years of experience teaching math.\n\n"

    "Your primary goal is to create a single, new math problem that a student will actually solve in a classroom.\n\n"

    "CRITICAL RULES:\n"
    "\t1. You MUST generate the problem itself, NOT instructions about how to create a problem.\n"
    "\t2. You MUST NOT generate meta-questions or meta-problems (e.g., \"Create a division problem...\").\n"
    "\t3. Base the problem *only* on the provided curriculum content.\n"
    "\t4. Always provide the output *only* in the requested format (Problem Type, Problem, Choices, Image).\n"
    "\t5. For problems involving shapes, only one shape may be included. "
    "This shape MUST be represented as SVG (Scalable Vector Graphics) code placed in the 'Image' field. "
    "If no image is needed, you MUST write 'N/A' in the 'Image' field.\n"
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
    "4. Image: [If an image is necessary for the problem, only one shape may be included. If not needed, write 'N/A']\n\n"
)

# Three-shot Examples
user_prompt += (
    "Example math problems\n\n"

    "Problem Type: multiple choices\n"
    "Problem: 좌표평면 위의 두 점 (1, -1), (2, 1)을 지나는 직선의 Y절편은? (난이도: 하)\n"
    "Choices: 1. -3, 2. -2, 3. -1, 4. 0, 5. 1\n"
    "Image: N/A\n\n"

    "Problem Type: multiple choices\n"
    "Problem: 두 자연수 a, b에 대하여 다항식 2x^2 + 9x + k가 (2x+a)(x+b)로 인수분해되도록 하는 실수 k의 최솟값은? (난이도: 중)\n"
    "Choices: 1. 1, 2. 4, 3. 7, 4. 10, 5. 13\n"
    "Image: N/A\n\n"

    "Problem Type: subjective\n"
    "Problem: p < q인 두 소수 p, q에 대하여 p^2q < n <= pq^2을 만족하는 자연수 n의 개수가 308개일 때, p+q를 구하시오. (난이도: 상)\n"
    "Choices: N/A\n"
    "Image: N/A\n\n"
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
        "content": system_prompt
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text", "text": user_prompt
                # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            """ 이미지에 대한 설명
            {
                \"type\": \"text\", \"text\":
            },
            """
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
print("Qwen3-VL-7B Instruct Output")
print(output_text)