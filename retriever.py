from pathlib import Path
import re
import os
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent
DOCUMENT_DIR = ROOT_DIR / "documents"
PERSIST_DIR = ROOT_DIR / "vectordb"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 필요한 디렉토리 생성
DOCUMENT_DIR.mkdir(exist_ok=True)
PERSIST_DIR.mkdir(exist_ok=True)

# Embedding 모델 초기화
embed_model = HuggingFaceEmbeddings(model_name=EMB_MODEL)

# Text Splitter 초기화
md_splitter = MarkdownHeaderTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

# 학년별 교육과정 파일 매핑
content_files = {
    "고등 1": "공통수학1_교육과정.md",
    "고등 2": "공통수학2_교육과정.md",
    "중등 1": "중등_13학년_교육과정.md",
    "중등 2": "중등_13학년_교육과정.md",
    "중등 3": "중등_13학년_교육과정.md",
    "초등 1": "초등_12학년_교육과정.md",
    "초등 2": "초등_12학년_교육과정.md",
    "초등 3": "초등_34학년_교육과정.md",
    "초등 4": "초등_34학년_교육과정.md",
    "초등 5": "초등_56학년_교육과정.md",
    "초등 6": "초등_56학년_교육과정.md"
}


def create_vectorstore(file: str, name: str, persist_directory: str = None) -> Chroma:
    """
    문서로부터 벡터스토어를 생성합니다.
    
    Args:
        file (str): 문서 파일명
        name (str): 벡터스토어 컬렉션 이름
        persist_directory (str, optional): 벡터스토어 저장 경로
        
    Returns:
        Chroma: 생성된 벡터스토어 인스턴스
    """
    # 파일 경로 처리
    file_path = DOCUMENT_DIR / file
    if not file_path.exists():
        # documents/ 내부에 없다면 현재 경로 기준으로도 한 번 시도
        alt_path = Path(file)
        if not alt_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file}")
        file_path = alt_path

    # 교육과정 파일 Load
    loader = UnstructuredMarkdownLoader(str(file_path))
    doc = loader.load()

    # Markdown Header 기반 Split
    header_splitted_docs = md_splitter.split_documents(doc)

    # RecursiveCharacterTextSplitter 기반 추가 Split
    docs = text_splitter.split_documents(header_splitted_docs)

    # persist_directory가 없으면 기본값 사용
    if persist_directory is None:
        persist_directory = PERSIST_DIR / name

    # persist_directory가 없으면 생성
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    # Embedding + VectorStore 생성
    vectorstore = Chroma.from_documents(
        collection_name=name,
        documents=docs,
        embedding=embed_model,
        persist_directory=str(persist_directory),
    )
    vectorstore.persist() # DB 저장
    return vectorstore


def parsing_user_input(user_input: str) -> tuple:
  """
  User_input을 파싱하여 과정/학년/학기/단원을 추출하고, RAG에 사용해야 할 Content 파일과 Query를 생성하는 함수

  어플에서 과정/학년/학기/단원을 입력받고,
  해당 정보를 Query로 구성하는 과정에서 4가지 중 하나가
  빠진 것이 없다는 가정하에 진행한다.

  Args:
      user_input (str): 사용자 입력 문자열  

  Returns:
      tuple: content_file, curriculum_query, course, grade, semester, unit
  """
  # input parsing
  course, grade, semester, unit = re.match(r"(초등|중등|고등)\s*(\d)학년\s*(\d)학기\s*(.+)", user_input.strip()).groups()

  # input으로부터 Content 파일 선택
  content = f"{course} {grade}"
  if content not in content_files:
    raise KeyError(f"해당 학년 자료가 없습니다: {content}")
  content_file = content_files[content]
  
  # 교육과정 총정리 파일에서 단원 선택
  content_file = str(DOCUMENT_DIR / content_files[content])
  curriculum_query = f"Let me know {course} {grade}grade {semester} semester's {unit} content in detail." 

  return content_file, curriculum_query, course, grade, semester, unit


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
  docs = retriever.get_relevant_documents(query)

  return docs
