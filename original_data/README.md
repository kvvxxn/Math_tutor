이 폴더는 src/math_tutor/preprocessing/preprocess_data.py 에서 사용할 원본 Dataset을 저장하는 폴더입니다.

폴더 구조는 다음과 같습니다.

- 각 폴더에는 개별 JSON 파일이 저장되어 있습니다.

```bash
original_data/
├── training/
│   └── labels/
│       ├── element3_problem/
│       ├── element3_solution/
│       ├── element4_problem/
│       ├── element4_solution/
│       ├── element5_problem/
│       ├── element5_solution/
│       ├── element6_problem/
│       ├── element6_solution/
│       ├── middle1_problem/
│       ├── middle1_solution/
│       ├── middle2_problem/
│       ├── middle2_solution/
│       ├── middle3_problem/
│       ├── middle3_solution/
│       ├── high_problem/
│       └── high_solution/
│
└── validation/
    └── labels/
        ├── element3_problem/
        ├── element3_solution/
        ├── element4_problem/
        ├── element4_solution/
        ├── element5_problem/
        ├── element5_solution/
        ├── element6_problem/
        ├── element6_solution/
        ├── middle1_problem/
        ├── middle1_solution/
        ├── middle2_problem/
        ├── middle2_solution/
        ├── middle3_problem/
        ├── middle3_solution/
        ├── high_problem/
        └── high_solution/
```