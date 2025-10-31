from pathlib import Path
from typing import Dict, List
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 프로젝트 루트 및 경로
ROOT_DIR = Path(__file__).parent
DOCUMENT_DIR = ROOT_DIR / "documents"
PERSIST_ROOT = ROOT_DIR / "vectordb"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 디렉토리 생성
DOCUMENT_DIR.mkdir(exist_ok=True)
PERSIST_ROOT.mkdir(exist_ok=True)

# 문서명 -> DB name 매핑
NAME_MAP: Dict[str, str] = {
    "공통수학1_교육과정": "Common_1",
    "공통수학2_교육과정": "Common_2",
    "중등_13학년_교육과정": "Middle_13", 
    "초등_12학년_교육과정": "Element_12",
    "초등_34학년_교육과정": "Element_34",
    "초등_56학년_교육과정": "Element_56",
}

# Splitter / Embedding (단일 파일용 create_vectorstore와 동일 구성)
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
    ("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")
])
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=80, separators=["\n\n", "\n"]
)
embed_model = HuggingFaceEmbeddings(model_name=EMB_MODEL)

def create_vectorstore_for_file(file_path: Path, name: str, persist_directory: Path = None) -> Chroma:
    """
    단일 파일용 create_vectorstore와 동일한 동작으로 벡터스토어 생성
    """
    if not file_path.exists():
        # documents/ 기준 실패 시, 현재 경로 기준 재시도
        alt_path = Path(file_path.name)
        if not alt_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        file_path = alt_path

    # Load
    loader = UnstructuredMarkdownLoader(str(file_path))
    doc = loader.load()

    # page content 추출 
    raw_text = doc[0].page_content

    # Split
    header_splitted_docs = md_splitter.split_text(raw_text)
    docs = text_splitter.split_documents(header_splitted_docs)

    # vectordb 폴더 생성
    if persist_directory is None:
        persist_directory = PERSIST_ROOT / name
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    # 벡터스토어 생성
    vectorstore = Chroma.from_documents(
        collection_name=name,
        documents=docs,
        embedding=embed_model,
        persist_directory=str(persist_directory),
    )
    return vectorstore

def build_all_vectorstores(directory: Path, skip: List[str] = None):
    """
    documents 폴더 내 모든 .md 파일에 대해 개별 벡터 DB 생성하는 함수
    """
    skip = skip or []
    md_files = sorted(directory.glob("*.md"))

    created = []
    for md in md_files:
        if md.name in skip:
            continue

        # 컬렉션 이름 결정: 위 Dictionary 참조
        collection_name = NAME_MAP.get(md.stem, md.stem)

        try:
            _ = create_vectorstore_for_file(
                file_path=md,
                name=collection_name,
                persist_directory=PERSIST_ROOT / collection_name
            )
            print(f"[OK] {md.name} -> {collection_name} @ {PERSIST_ROOT / collection_name}")
            created.append(md.name)
        except Exception as e:
            print(f"[FAIL] {md.name}: {e}")

    print(f"\nCreated {len(created)} Chroma DB(s).")
    for c in created:
        print(" -", c)

def main():
    skip = ["교육과정총정리.md"]
    build_all_vectorstores(DOCUMENT_DIR, skip=skip)

if __name__ == "__main__":
    main()
