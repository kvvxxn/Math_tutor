#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data/
 ├─ Training/
 │   ├─ images/   (PNG들)
 │   └─ labels/   (문제 JSON, 해설 JSON_A)
 └─ Validation/
     ├─ images/
     └─ labels/

출력:
Data/
 ├─ train_image_text.jsonl
 ├─ train_image_only.jsonl
 ├─ train_text_only.jsonl
 ├─ train_mixed.jsonl
 ├─ val_image_text.jsonl
 ├─ val_image_only.jsonl
 ├─ val_text_only.jsonl
 └─ val_mixed.jsonl
"""
import json, re, random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ---------- 공통 유틸 ----------
def read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def normalize_question_text(prob: Dict[str, Any]) -> str:
    ocr = prob.get("OCR_info") or []
    if ocr:
        qb = ocr[0].get("question_bbox") or []
        for b in qb:
            if b.get("type") == "paragraph" and b.get("text"):
                return b["text"].strip()
        qt = ocr[0].get("question_text")
        if qt:
            return qt.strip()
    return ""

CIRCLED_MAP = {
    "①":"A","②":"B","③":"C","④":"D","⑤":"E","⑥":"F","⑦":"G","⑧":"H","⑨":"I","⑩":"J",
    "⑪":"K","⑫":"L","⑬":"M","⑭":"N","⑮":"O","⑯":"P","⑰":"Q","⑱":"R","⑲":"S","⑳":"T",
    "1":"A","2":"B","3":"C","4":"D","5":"E","6":"F","7":"G","8":"H","9":"I","10":"J"
}

def map_choice_symbol(sym: str) -> Optional[str]:
    sym = (sym or "").strip()
    m = re.search(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|(?<!\d)(10|[1-9])(?!\d)", sym)
    if not m:
        return None
    token = m.group(0)
    return CIRCLED_MAP.get(token)

def extract_choice_from_answer_bbox(bboxes: List[Dict[str, Any]]) -> Optional[str]:
    prefer = [bb for bb in (bboxes or []) if bb.get("type") == "answer"]
    scan = prefer if prefer else (bboxes or [])
    for bb in scan:
        ch = map_choice_symbol(bb.get("text") or "")
        if ch:
            return ch
    return None

def extract_value_from_text(txt: str) -> Optional[str]:
    if not txt:
        return None
    s = txt.replace("‒", "-").replace("−", "-")
    pats = [
        r"정답(?:은|:)?\s*([\-]?\d+(?:\.\d+)?)",
        r"(?:값|최댓값|최소값|최대값|최솟값|합|개수|경우의 수|확률)\s*(?:은|는|:)\s*([\-]?\d+(?:\.\d+)?)",
        r"=\s*([\-]?\d+(?:\.\d+)?)\s*(?:이다|입니다|임|임\.)?"
    ]
    for p in pats:
        m = re.search(p, s)
        if m:
            return m.group(1)
    nums = re.findall(r"[\-]?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else None

def extract_multi_part_answers(answer_text: str, answer_bbox_text: Optional[str]=None) -> List[Tuple[int,str]]:
    src = (answer_bbox_text or "").strip() or (answer_text or "")
    if not src:
        return []
    src = src.replace("‒", "-").replace("−", "-")
    parts = re.split(r"\((\d+)\)", src)
    out: List[Tuple[int,str]] = []
    i = 1
    while i < len(parts):
        try:
            idx = int(parts[i])
        except:
            i += 1; continue
        seg = parts[i+1] if i+1 < len(parts) else ""
        m = re.findall(r"=\s*([^=,\n;]+)", seg)
        if m:
            candidate = m[-1].strip()
            candidate = re.sub(r"[^\w\-\.\^\\\/]+$", "", candidate)
            out.append((idx, candidate))
        else:
            nums = re.findall(r"[\-]?\d+(?:\.\d+)?", seg)
            if nums:
                out.append((idx, nums[-1]))
        i += 2
    return sorted(out, key=lambda x: x[0])

def build_user_message(qtext: str, mode: str) -> str:
    """
    mode: 'image_text' | 'image_only' | 'text_only'
    """
    rule = "[규칙]\n- 계산만 하고 최종값만 출력.\n- 출력 형식: \"정답: <값>\""
    if mode == "image_text":
        body = qtext or "이미지에 제시된 문제를 풀어라."
        return f"<image>\n[문제]\n{body}\n\n{rule}"
    elif mode == "image_only":
        return f"<image>\n[문제]\n이미지에 제시된 문제를 풀어라.\n\n{rule}"
    elif mode == "text_only":
        body = qtext or "(문제 텍스트 미검출)"
        return f"[문제]\n{body}\n\n{rule}"
    else:
        raise ValueError("unknown mode")

def build_assistant_answer(ans_choice: Optional[str], ans_value: Optional[str], multi_parts: List[Tuple[int,str]]) -> str:
    if multi_parts:
        joined = "; ".join(f"{idx}) {val}" for idx, val in multi_parts)
        return f"정답: {joined}"
    if ans_value is not None:
        return f"정답: {ans_value}"
    if ans_choice is not None:
        return f"정답: {ans_choice}"
    return "정답: "

def make_samples_for_split(split_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """split_dir = Data/Training or Data/Validation"""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    # 라벨 파일 수집: 문제/해설 매칭
    problems: Dict[str, Path] = {}
    solutions: Dict[str, Path] = {}
    for p in labels_dir.glob("*.json"):
        stem = p.stem
        if stem.endswith("_A"):
            solutions[stem[:-2]] = p
        else:
            problems[stem] = p

    image_text, image_only, text_only = [], [], []

    for pid, pfile in problems.items():
        sfile = solutions.get(pid)
        if not sfile:
            continue
        prob = read_json(pfile); sol = read_json(sfile)
        if not prob or not sol:
            continue

        img_name = prob.get("question_filename")
        img_path = images_dir / img_name if img_name else None
        has_img = img_path is not None and img_path.exists()

        # question text
        qtext = normalize_question_text(prob)

        # answer parse
        ai = (sol.get("answer_info") or [None])[0] or {}
        ans_text = ai.get("answer_text") or ""
        ans_bbox = ai.get("answer_bbox") or []
        ans_choice = extract_choice_from_answer_bbox(ans_bbox) if ans_bbox else None
        answer_bbox_text = None
        for bb in ans_bbox:
            if bb.get("type") == "answer" and bb.get("text"):
                answer_bbox_text = bb["text"]; break
        multi = extract_multi_part_answers(ans_text, answer_bbox_text)
        value = extract_value_from_text(ans_text) if not multi else None

        # 1) Image+Text (이미지 있고 텍스트 있는 경우만)
        if has_img and qtext:
            image_text.append({
                "id": prob.get("id"),
                "images": [str(img_path)],
                "conversations": [
                    {"from":"user","value":build_user_message(qtext, "image_text")},
                    {"from":"assistant","value":build_assistant_answer(ans_choice, value, multi)}
                ],
                "meta": prob.get("question_info",[{}])[0]
            })

        # 2) ImageOnly (이미지만)
        if has_img:
            image_only.append({
                "id": prob.get("id"),
                "images": [str(img_path)],
                "conversations": [
                    {"from":"user","value":build_user_message(qtext, "image_only")},
                    {"from":"assistant","value":build_assistant_answer(ans_choice, value, multi)}
                ],
                "meta": prob.get("question_info",[{}])[0]
            })

        # 3) TextOnly (텍스트만)
        if qtext:
            text_only.append({
                "id": prob.get("id"),
                "images": [],  # 멀티모달 모델도 images 없는 샘플을 학습에 넣어도 됨
                "conversations": [
                    {"from":"user","value":build_user_message(qtext, "text_only")},
                    {"from":"assistant","value":build_assistant_answer(ans_choice, value, multi)}
                ],
                "meta": prob.get("question_info",[{}])[0]
            })

    return {"image_text": image_text, "image_only": image_only, "text_only": text_only}

def write_jsonl(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sample_mixed(splits: Dict[str, List[Dict[str, Any]]], mix: Tuple[float,float,float]) -> List[Dict[str, Any]]:
    it, io, to = splits["image_text"], splits["image_only"], splits["text_only"]
    total = len(it) + len(io) + len(to)
    if total == 0:
        return []
    n_it = int(round(mix[0] * total))
    n_io = int(round(mix[1] * total))
    n_to = int(round(mix[2] * total))
    # 조정
    def take(lst, n):
        if n <= 0: return []
        if n >= len(lst): return lst
        return random.sample(lst, n)
    mixed = take(it, n_it) + take(io, n_io) + take(to, n_to)
    random.shuffle(mixed)
    return mixed

# ---------- 엔트리 ----------
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("Data"))
    parser.add_argument("--train_mix", type=str, default="0.65,0.25,0.10",
                        help="ImageText,ImageOnly,TextOnly ratio")
    parser.add_argument("--val_mix", type=str, default="0.65,0.25,0.10")
    args = parser.parse_args()

    random.seed(42)

    # ratio 파싱
    def parse_ratio(s: str):
        a,b,c = [float(x) for x in s.split(",")]
        ssum = a+b+c
        if ssum <= 0: return (0.65,0.25,0.10)
        return (a/ssum, b/ssum, c/ssum)

    train_ratio = parse_ratio(args.train_mix)
    val_ratio = parse_ratio(args.val_mix)

    # Training
    train_splits = make_samples_for_split(args.data_root / "Training")
    write_jsonl(train_splits["image_text"], args.data_root / "train_image_text.jsonl")
    write_jsonl(train_splits["image_only"], args.data_root / "train_image_only.jsonl")
    write_jsonl(train_splits["text_only"], args.data_root / "train_text_only.jsonl")
    write_jsonl(sample_mixed(train_splits, train_ratio), args.data_root / "train_mixed.jsonl")

    # Validation
    val_splits = make_samples_for_split(args.data_root / "Validation")
    write_jsonl(val_splits["image_text"], args.data_root / "val_image_text.jsonl")
    write_jsonl(val_splits["image_only"], args.data_root / "val_image_only.jsonl")
    write_jsonl(val_splits["text_only"], args.data_root / "val_text_only.jsonl")
    write_jsonl(sample_mixed(val_splits, val_ratio), args.data_root / "val_mixed.jsonl")

    # 로그
    def stat(name, s):
        print(f"{name}: IT={len(s['image_text'])}, IO={len(s['image_only'])}, TO={len(s['text_only'])}")

    print("DONE")
    stat("Training", train_splits)
    stat("Validation", val_splits)
