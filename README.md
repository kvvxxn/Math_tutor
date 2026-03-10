# Math_tutor

Repository for math problem solver and generator using LLM

**Problem Generator**

- `Input Format`: [학교급] [학년] [학기] (예: 초등 6학년 1학기)

- `Scope`: 초등 전체, 중등 전체, 고등 1학년

- Trained by AI Hub dataset

**Problem Solver**

- 학생이 이해할 수 있는 수준의 사고 흐름(Chain-of-Thought)을 제공

- 논리적인 답변을 생성하도록 Prompt Engineering으로 유도


**Model**: QWEN3-VL-8B-INSTRUCT

**Dataset**: [AI Hub 수학 과목 문제 생성 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71718)

- Used for training Generator
