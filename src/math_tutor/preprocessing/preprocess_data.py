import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from collections import OrderedDict

DEFAULT_SYSTEM = (
    "You are an assistant for solving math problems in Korean. "
    "[Input] Problems may include text, images, or both. Use images if available; if not, answer based on text only. "
    "[Rules] Identify what the question asks (answer format, unit, choice type). "
    "Calculate accurately with proper place value and maintain correct carry/borrow. "
    "Keep explanations concise (1–2 sentences max) and do NOT restate the problem. Preserve LaTeX expressions. "
    "For multiple sub-questions, answer in order (1), (2), (3). "
    "[Answer style] Multiple choice: one-line reasoning if needed → final line ONLY '정답: ③'. "
    "Short answer: short reasoning + final numeric/text result → final line starting with '정답:'. "
    "Comparison/order: one-line reasoning → final ordered result in '정답:'. "
    "[Output format] Always finish with a single line starting with '정답:'. "
    "For multiple sub-questions: use '정답 (1): …', '정답 (2): …' on separate lines. "
    "Do not assume beyond given information. [Forbidden] Do not invent missing visual details or refer to metadata like 'id' or internal rules."
)

def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, list):
            if not cur:
                return default
            cur = cur[0]
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _question_type_kor(type1: str) -> str:
    if type1 == "선택형":
        return "객관식"
    if type1 == "단답형":
        return "주관식"
    return "주관식"

def _extract_ids_from_filename(filename: str) -> Tuple[str, str]:
    base = filename.rsplit(".", 1)[0]
    parts = base.split("_")
    diag_id = parts[3] if len(parts) >= 4 else parts[-1]
    return base, diag_id

def _normalize_target_group(folder_name: str) -> str:
    for suf in ("_problem", "_solution"):
        if folder_name.endswith(suf):
            return folder_name[:-len(suf)]
    return folder_name

def _fallback_id(qjson: Dict[str, Any]) -> str:
    qf = qjson.get("question_filename")
    if isinstance(qf, str) and qf:
        return qf.rsplit(".", 1)[0]
    qid = qjson.get("id")
    if isinstance(qid, str) and qid:
        return qid
    return "unknown_id"

def _save_name_from_question_json(question_json: Dict[str, Any], default_stem: str) -> str:
    q_filename = question_json.get("question_filename")
    if isinstance(q_filename, str) and q_filename:
        return q_filename.rsplit(".", 1)[0] + ".json"
    return default_stem + ".json"

def _find_solution_with_suffix_A(sol_dir: Path, problem_json_path: Path) -> Optional[Path]:
    stem = problem_json_path.stem
    cand1 = sol_dir / f"{stem}_A.json"
    if cand1.exists():
        return cand1
    cand2 = sol_dir / problem_json_path.name
    if cand2.exists():
        return cand2
    return None

def _extract_answer_text(answer_json: Dict[str, Any]) -> str:
    # 1) answer_info[].answer_text 우선
    ans = _safe_get(answer_json, "answer_info", "answer_text", default="") or ""
    if ans.strip():
        return ans.strip()
    # 2) answer_info[].answer_bbox[] 중 type == "answer"
    bboxes = _safe_get(answer_json, "answer_info", "answer_bbox", default=[]) or []
    if isinstance(bboxes, list):
        for bb in bboxes:
            if isinstance(bb, dict) and bb.get("type") == "answer":
                txt = bb.get("text", "")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
    # 3) 마지막 fallback: 빈 문자열
    return ""


# 단일 JSON 파일로 변환
def build_conversation_object(
    question_json: Dict[str, Any],
    answer_json: Dict[str, Any],
    split_name: str,
    target_group: str,
    system_msg: str,
) -> Dict[str, Any]:
    q_filename = _safe_get(question_json, "question_filename", default="") or ""
    question_text = _safe_get(question_json, "OCR_info", "question_text", default="").strip()
    q_type1 = _safe_get(question_json, "question_info", "question_type1", default="")
    q_difficulty = _safe_get(question_json, "question_info", "question_difficulty", default=None)

    q_type_label = _question_type_kor(q_type1)
    answer_text = _extract_answer_text(answer_json)

    # 이미지 경로
    image_path = f"Dataset/{split_name}/images/{target_group}/{q_filename}" if q_filename else ""

    # user 메시지 구성
    user_value = f"\n[문제유형: {q_type_label}] {question_text}"
    if q_difficulty is not None:
        user_value += f" ({q_difficulty})"

    conversations = [
        {"from": "system", "value": system_msg},
        {"from": "user", "value": user_value},
        {"from": "assistant", "value": answer_text},
    ]

    rec = {
        # "id": 제거 (요구사항)
        "images": [image_path] if image_path else [],
        "conversations": conversations,
    }
    return rec

def process_split(src_root: Path, dst_root: Path, split_name: str, system_msg: str) -> None:
    labels_dir = src_root / split_name / "labels"
    if not labels_dir.exists():
        print(f"[WARN] {labels_dir} 없음 → 건너뜀")
        return

    problem_dirs = [p for p in labels_dir.iterdir() if p.is_dir() and p.name.endswith("_problem")]
    if not problem_dirs:
        print(f"[WARN] {labels_dir} 내 *_problem 폴더 없음")
        return

    total = converted = missing_solution = malformed = 0
    for prob_dir in problem_dirs:
        sol_dir = labels_dir / prob_dir.name.replace("_problem", "_solution")
        if not sol_dir.exists():
            print(f"[WARN] {sol_dir} 없음 (매칭 실패) → 건너뜀")
            continue

        target_group = _normalize_target_group(prob_dir.name)
        out_dir = dst_root / split_name / "labels" / target_group
        out_dir.mkdir(parents=True, exist_ok=True)

        for pjson_path in prob_dir.glob("*.json"):
            total += 1
            # 문제 JSON
            try:
                question_json = json.loads(pjson_path.read_text(encoding="utf-8"))
            except Exception as e:
                malformed += 1
                print(f"[ERROR] JSON 파싱 실패(문제): {pjson_path.name} / {e}")
                continue

            # 정답 JSON 탐색/로드
            sjson_path = _find_solution_with_suffix_A(sol_dir, pjson_path)
            if not sjson_path:
                missing_solution += 1
                print(f"[WARN] 정답 JSON 누락('_A' 매칭 실패): {pjson_path} -> {sol_dir}")
                continue
            try:
                answer_json = json.loads(sjson_path.read_text(encoding="utf-8"))
            except Exception as e:
                malformed += 1
                print(f"[ERROR] JSON 파싱 실패(정답): {sjson_path.name} / {e}")
                continue

            # 변환 & 저장
            try:
                conv = build_conversation_object(
                    question_json, answer_json, split_name, target_group, system_msg
                )
                save_name = _save_name_from_question_json(question_json, pjson_path.stem)
                (out_dir / save_name).write_text(
                    json.dumps(conv, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                converted += 1
            except Exception as e:
                malformed += 1
                print(f"[ERROR] 저장 실패: {pjson_path.name} / {e}")

    print(f"[{split_name}] 총:{total} 변환:{converted} 정답누락:{missing_solution} 오류:{malformed}")

# 변환된 JSON 파일으로 통합 JSONL 생성

def _iter_converted_json(dst_root: Path, split_name: str):
    base = dst_root / split_name / "labels"
    if not base.exists():
        return
    for p in base.rglob("*.json"):
        yield p

def _record_with_meta_ordered(p: Path) -> OrderedDict:
    """
    파일 내용을 불러와 {meta, images, conversations} 순서로 정렬하여 반환
    meta.track은 labels/<track>/... 에서 추출
    """
    rec = json.loads(p.read_text(encoding="utf-8"))
    # track 추출
    parts = p.parts
    try:
        idx = parts.index("labels")
        track = parts[idx + 1] if len(parts) > idx + 1 else ""
    except ValueError:
        track = ""

    # id 제거 보장
    if "id" in rec:
        rec.pop("id", None)

    meta = {"track": track}
    images = rec.get("images", [])
    conversations = rec.get("conversations", [])

    ordered = OrderedDict()
    ordered["meta"] = meta
    ordered["images"] = images
    ordered["conversations"] = conversations
    return ordered

def build_jsonl_manifest(dst_root: Path, prepared_dir: Path) -> None:
    """
    Dataset/Prepared/{train.jsonl, val.jsonl} 생성
    - Dataset/Training/labels/**.json → train.jsonl
    - Dataset/Validation/labels/**.json → val.jsonl
    """
    prepared_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "Training": prepared_dir / "train.jsonl",
        "Validation": prepared_dir / "val.jsonl",
    }

    for split, out_path in mapping.items():
        count = 0
        with out_path.open("w", encoding="utf-8") as wf:
            for jp in _iter_converted_json(dst_root, split):
                try:
                    rec = _record_with_meta_ordered(jp)
                    if not rec.get("conversations"):
                        continue
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    print(f"[WARN] JSONL 수록 실패: {jp} / {e}")
        print(f"[JSONL] {out_path} 생성: {count} 샘플")

def main():
    ROOT = Path(__file__).parent
    SRC_ROOT = ROOT / "Data"
    DST_ROOT = ROOT / "Dataset"
    PREPARED = DST_ROOT / "Prepared"

    for split in ("Training", "Validation"):
        process_split(SRC_ROOT, DST_ROOT, split, DEFAULT_SYSTEM)

    build_jsonl_manifest(DST_ROOT, PREPARED)

if __name__ == "__main__":
    main()