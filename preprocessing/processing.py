import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

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
    """
    'M1_2_07_00043_12793.png' ->
      base_id: 'M1_2_07_00043_12793'
      diag_id: '00043'  (4번째 토큰)
    """
    base = filename.rsplit(".", 1)[0]
    parts = base.split("_")
    diag_id = parts[3] if len(parts) >= 4 else parts[-1]
    return base, diag_id

def _normalize_target_group(folder_name: str) -> str:
    """
    'element3_problem', 'middle1_solution', 'high_problem' 형태라고 가정.
    접미사 제거하여 타깃 그룹명: element3, middle1, high
    """
    for suf in ("_problem", "_solution"):
        if folder_name.endswith(suf):
            return folder_name[:-len(suf)]
    return folder_name

# JSON 변환 오브젝트 생성
def build_conversation_object(
    question_json: Dict[str, Any],
    answer_json: Dict[str, Any],
    split_name: str,
    target_group: str,
) -> Dict[str, Any]:
    q_filename = _safe_get(question_json, "question_filename", default="") or ""
    question_text = _safe_get(question_json, "OCR_info", "question_text", default="").strip()
    q_type1 = _safe_get(question_json, "question_info", "question_type1", default="")
    q_difficulty = _safe_get(question_json, "question_info", "question_difficulty", default=None)

    q_type_label = _question_type_kor(q_type1)
    answer_text = _safe_get(answer_json, "answer_info", "answer_text", default="") or ""

    base_id, _ = ("", "")
    if q_filename:
        base_id, _ = _extract_ids_from_filename(q_filename)

    user_value = f"\n[문제유형: {q_type_label}] {question_text}"
    if q_difficulty is not None:
        user_value += f" ({q_difficulty})"

    # 이미지 경로는 Dataset 폴더 기준
    image_path = f"Dataset/{split_name}/images/{target_group}/{q_filename}" if q_filename else ""

    return {
        "id": base_id if base_id else _fallback_id(question_json),
        "images": [image_path] if image_path else [],
        "conversations": [
            {"from": "user", "value": user_value},
            {"from": "assistant", "value": answer_text}
        ]
    }

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
    """
    기본 규칙: 문제 파일명이 P3_1_01_00040_00469.json 이면
    해설은 P3_1_01_00040_00469_A.json (동일 stem + '_A')
    1) <stem>_A.json 우선 시도
    2) 동일 파일명도 보조 시도 (혹시 일부가 _A 없이 같은 이름일 수 있어서)
    """
    stem = problem_json_path.stem  
    cand1 = sol_dir / f"{stem}_A.json"
    if cand1.exists():
        return cand1
    cand2 = sol_dir / problem_json_path.name
    if cand2.exists():
        return cand2
    return None

def process_split(src_root: Path, dst_root: Path, split_name: str) -> None:
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

            # Problem JSON 파일 읽기
            try:
                question_json = json.loads(pjson_path.read_text(encoding="utf-8"))
            except Exception as e:
                malformed += 1
                print(f"[ERROR] JSON 파싱 실패(문제): {pjson_path.name} / {e}")
                continue

            # Solution JSON 파일명 지정
            sjson_path = _find_solution_with_suffix_A(sol_dir, pjson_path)
            if not sjson_path:
                missing_solution += 1
                print(f"[WARN] 정답 JSON 누락(패턴 '_A' 매칭 실패): {pjson_path} -> {sol_dir}")
                continue

            # Solution JSON 파일 읽기
            try:
                answer_json = json.loads(sjson_path.read_text(encoding="utf-8"))
            except Exception as e:
                malformed += 1
                print(f"[ERROR] JSON 파싱 실패(정답): {sjson_path.name} / {e}")
                continue

            try:
                conv = build_conversation_object(
                    question_json, answer_json, split_name, target_group
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

# 변환된 JSON 파일으로부터 통합 JSONL 생성

def _iter_converted_json(dst_root: Path, split_name: str):
    """
    Dataset/<split>/labels/<track>/*.json 을 재귀 탐색해 yield
    """
    base = dst_root / split_name / "labels"
    if not base.exists():
        return
    for p in base.rglob("*.json"):
        yield p

def _record_with_meta(p: Path) -> Dict[str, Any]:
    """
    파일 내용을 불러와, track 메타를 추가해 반환
    """
    rec = json.loads(p.read_text(encoding="utf-8"))
    # track: labels/<track>/... 에서 추출
    parts = p.parts
    try:
        idx = parts.index("labels")
        track = parts[idx+1] if len(parts) > idx+1 else ""
    except ValueError:
        track = ""
    rec.setdefault("meta", {})["track"] = track
    return rec

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
                    rec = _record_with_meta(jp)
                    # sanity: 필수 키가 없으면 스킵
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
        process_split(SRC_ROOT, DST_ROOT, split)

    # 변환 완료 후 통합 JSONL 생성
    build_jsonl_manifest(DST_ROOT, PREPARED)

if __name__ == "__main__":
    main()