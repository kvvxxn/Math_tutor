이 폴더는 src/math_tutor/training/finetune_solver.py 에서 사용할 Training / Validation Dataset을 저장하는 폴더입니다.

폴더 구조는 다음과 같습니다.

- images의 각 폴더에는 문제에 해당하는 이미지 데이터가 저장되어 있습니다.

- labels는 scr/math_tutor/preprocessing/preproecss_data.py의 실행 결과로 생긴 변환된 JSON 파일이 저장되는 폴더입니다.
  
- JSONL 파일은 labels 폴더 내 변환된 JSON 파일을 이용하여 생성된 파일입니다.

```bash
original_data/
├── training/
│   └── images/
│       ├── element3/
│       ├── element4/
│       ├── element5/
│       ├── element6/
│       ├── middle1/
│       ├── middle2/
│       ├── middle3/
│       ├── high/
│       └── high_solution/
│   └── labels/
│
├── validation/
│   └── images/
│       ├── element3/
│       ├── element4/
│       ├── element5/
│       ├── element6/
│       ├── middle1/
│       ├── middle2/
│       ├── middle3/
│       ├── high/
│       └── high_solution/
│   └── labels/
└── prepared/
│   └── labels/
│       ├── train.jsonl
│       └── val.jsonl
```