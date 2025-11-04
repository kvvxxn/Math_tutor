import json
from pathlib import Path
from typing import Dict, Any, Tuple

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
    원본 폴더가 이미 'element3_problem', 'middle1_solution', 'high_problem' 형태라고 가정.
    접미사만 제거하여 타깃 그룹명으로 사용: element3, middle1, high
    """
    for suf in ("_problem", "_solution"):
        if folder_name.endswith(suf):
            return folder_name[:-len(suf)]
    return folder_name 

# JSON Conversion file 생성
def build_conversation_object(question_json: Dict[str, Any],
                              answer_json: Dict[str, Any]) -> Dict[str, Any]:
    q_filename = _safe_get(question_json, "question_filename", default="") or ""
    question_text = _safe_get(question_json, "OCR_info", "question_text", default="").strip()
    q_type1 = _safe_get(question_json, "question_info", "question_type1", default="")
    q_difficulty = _safe_get(question_json, "question_info", "question_difficulty", default=None)

    q_type_label = _question_type_kor(q_type1)
    answer_text = _safe_get(answer_json, "answer_info", "answer_text", default="") or ""

    base_id, diag_id = ("", "")
    if q_filename:
        base_id, diag_id = _extract_ids_from_filename(q_filename)

    user_value = f"\n[문제유형: {q_type_label}] {question_text}"
    if q_difficulty is not None:
        user_value += f" ({q_difficulty})"

    return {
        "id": base_id if base_id else _fallback_id(question_json),
        "images": [f"diagrams/{diag_id}.png"] if diag_id else [],
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

        if not out_dir.exists():
            print(f"[WARN] 출력 폴더 없음: {out_dir} → 스킵")
            continue

        for pjson_path in prob_dir.glob("*.json"):
            total += 1
            sjson_path = sol_dir / pjson_path.name
            if not sjson_path.exists():
                missing_solution += 1
                print(f"[WARN] 정답 JSON 누락: {sjson_path}")
                continue

            try:
                question_json = json.loads(pjson_path.read_text(encoding="utf-8"))
                answer_json = json.loads(sjson_path.read_text(encoding="utf-8"))
            except Exception as e:
                malformed += 1
                print(f"[ERROR] JSON 파싱 실패: {pjson_path.name} / {e}")
                continue

            try:
                conv = build_conversation_object(question_json, answer_json)
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

def main():
    SRC_ROOT = Path("Data")
    DST_ROOT = Path("Dataset")
    for split in ("Training", "Validation"):
        process_split(SRC_ROOT, DST_ROOT, split)

if __name__ == "__main__":
    main()