from pathlib import Path
import re
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

DOCUMENT_DIR = Path("./documents") 
PERSIST_DIR = str("./vectordb")   
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

content_files = {
  "고등 1":"공통수학1_교육과정.md",
  "고등 2": "공통수학2_교육과정.md",
  "중등 1":"중등_13학년_교육과정.md",
  "중등 2":"중등_13학년_교육과정.md",
  "중등 3":"중등_13학년_교육과정.md",
  "초등 1":"초등_12학년_교육과정.md",
  "초등 2":"초등_12학년_교육과정.md",
  "초등 3":"초등_34학년_교육과정.md",
  "초등 4":"초등_34학년_교육과정.md",
  "초등 5":"초등_56학년_교육과정.md",
  "초등 6":"초등_56학년_교육과정.md"
}
ciriculum_file = "교육과정총정리.md"

# Markdown Header 기반 Splitter 정의
headers = [("#","h1"),("##","h2"),("###","h3"),("####","h4")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

# Markdown 문서 내에서 RecursveCharacterTextSplitter 정의
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    separators=["\n\n", "\n"]
)
embed_model = HuggingFaceEmbeddings(model_name=EMB_MODEL)

def parsing_user_input(user_input: str) -> tuple:
  """
  어플에서 과정/학년/학기/단원을 입력받고,
  해당 정보를 Query로 구성하는 과정에서 4가지 중 하나가
  빠진 것이 없다는 가정하에 진행
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
  ciriculum_query = f"Let me know {course} {grade}grade {semester} semester's {unit} content in detail." 

  return content_file, ciriculum_query, course, grade, semester, unit

def create_vectorstore(file: str, name: str) -> tuple:
  if not os.path.exists(file_path):
    # documents/ 내부에 없다면 현재 경로 기준으로도 한 번 시도
    alt_path = str(Path(file_path))
    if not os.path.exists(alt_path):
      raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    file_path = alt_path

  # 교육과정 파일 Load
  loader = UnstructuredMarkdownLoader(file)
  doc = loader.load()

  # Markdown Header 기반 Split
  header_splitted_docs = md_splitter.split_documents(doc)

  # RecursiveCharacterTextSplitter 기반 추가 Split
  docs = text_splitter.split_documents(header_splitted_docs)

  # Embedding + VectorStore 생성
  vectorstore = Chroma.from_documents(
      collection_name=name,
      documents=docs,
      embedding=embed_model,
      persist_directory=PERSIST_DIR,
  )
  vectorstore.persist() # DB 저장

  return vectorstore

def retrieve(retriever, query: str, k: int) -> list[Document]:
  retriever.search_kwargs = {"k": k}
  docs = retriever.get_relevant_documents(query)

  return docs