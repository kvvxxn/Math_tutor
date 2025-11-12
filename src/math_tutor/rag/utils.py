from pathlib import Path
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from math_tutor.config.path import DOCUMENT_DIR, VECTORDB_DIR, ROOT_DIR

EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 필요한 디렉토리 생성
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

# Embedding 모델 초기화
embed_model = HuggingFaceEmbeddings(model_name=EMB_MODEL)

# Text Splitter 초기화
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

DB_MAP = {
    "고등 1 1": "Common_1",
    "고등 1 2": "Common_2",
    "중등 1 1": "Middle_13",
    "중등 1 2": "Middle_13",
    "중등 1 3": "Middle_13",
    "중등 2 1": "Middle_13",
    "중등 2 2": "Middle_13",
    "중등 2 3": "Middle_13",
    "중등 3 1": "Middle_13",
    "중등 3 2": "Middle_13",
    "중등 3 3": "Middle_13",
    "초등 1 1": "Element_12",
    "초등 1 2": "Element_12",
    "초등 2 1": "Element_12",
    "초등 2 2": "Element_12",
    "초등 3 1": "Element_34",
    "초등 3 2": "Element_34",
    "초등 4 1": "Element_34",
    "초등 4 2": "Element_34",
    "초등 5 1": "Element_56",
    "초등 5 2": "Element_56",
    "초등 6 1": "Element_56",
    "초등 6 2": "Element_56",
}


# TODO: 사용자 Input과 DB 연결


def parsing_user_input(user_input: str) -> tuple:
    """
    User_input을 파싱하여 과정/학년/학기/단원을 추출하고, RAG에 사용해야 할 Vector DB를 반환하는 함수

    어플에서 과정/학년/학기/단원을 입력받고,
    해당 정보를 Query로 구성하는 과정에서 4가지 중 하나가
    빠진 것이 없다는 가정하에 진행한다.

    Args:
        user_input (str): 사용자 입력 문자열  

    Returns:
        tuple: vectordb, course, grade, semester, unit
    """
    # input parsing
    course, grade, semester, unit = re.match(r"(초등|중등|고등)\s*(\d)학년\s*(\d)학기\s*(.+)", user_input.strip()).groups()

    # Vector DB 선택
    db_key = f"{course} {grade} {semester}"
    vectordb = Chroma(
        collection_name=DB_MAP[db_key],
        embedding_function=embed_model,
        VECTORDB_DIRectory=str(VECTORDB_DIR / DB_MAP[db_key])
    )

    return vectordb, course, grade, semester, unit


def retrieve(retriever, query: str, k: int) -> list[Document]:
  """
  query에 대해 retriever를 사용하여 k개의 관련 문서를 검색하는 함수

  Args: 
      retriever: 벡터스토어의 retriever 인스턴스
      query (str): 검색할 쿼리 문자열
      k (int): 검색할 문서의 개수

  Returns:
      list[Document]: 검색된 문서들의 리스트
  """
  retriever.search_kwargs = {"k": k}
  docs = retriever.invoke(query)

  return docs
