# CARE-PD Reproduction Guide (RF vs Official)

이 문서는 `CARE-PD` 데이터 기준으로 아래를 재현하기 위한 실행 가이드입니다.

- RF baseline (`src/research_automation/pipeline/gait_baseline.py`)
- CARE-PD official code (`data/datasets/CARE-PD-code`)
- 비교 지표: `Macro-F1` (within/LODO)

## 1) Environment

- Python: `3.12`
- venv: `web/.venv312`
- 공식 코드 위치: `data/datasets/CARE-PD-code`
- 데이터 위치: `data/datasets/CARE-PD`

## 2) Dataset 준비

### 다운로드

- CARE-PD dataset: `https://huggingface.co/datasets/vida-adl/CARE-PD/tree/main`
- CARE-PD official code: `https://github.com/TaatiTeam/CARE-PD`

### 배치

- `data/datasets/CARE-PD/` 아래에 `.pkl` 및 `folds/` 배치
- `data/datasets/CARE-PD-code/assets/datasets` → `data/datasets/CARE-PD` 심볼릭 링크

## 3) 전처리

공식 코드 입력 포맷(h36m) 생성:

```bash
cd data/datasets/CARE-PD-code
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE ./../../../web/.venv312/bin/python data/preprocessing/smpl2h36m.py -db BMCLab
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE ./../../../web/.venv312/bin/python data/preprocessing/smpl2h36m.py -db 3DGait
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE ./../../../web/.venv312/bin/python data/preprocessing/smpl2h36m.py -db PD-GaM
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE ./../../../web/.venv312/bin/python data/preprocessing/smpl2h36m.py -db T-SDU-PD
```

## 4) RF baseline 실행

```bash
./web/.venv312/bin/python src/research_automation/pipeline/gait_baseline.py data/datasets/CARE-PD --protocol literature --method rf
```

출력:

- Within(LOSO): dataset별 Macro-F1 + mean
- LODO: dataset별 Macro-F1 + mean

## 5) Official 코드 실행 (문헌 코드 직접)

### 주의사항

- official 코드는 기본적으로 CUDA 가정이 강함
- CPU 실행 환경에서는 시간이 오래 걸림 (수십 분~수 시간/실험)
- `WANDB_MODE=disabled` 권장

### 예시: LODO (BMCLab target, POTR)

```bash
cd data/datasets/CARE-PD-code
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled WANDB_SILENT=true \
./../../../web/.venv312/bin/python eval_only.py \
  --backbone potr \
  --config BMCLab.json \
  --hypertune 0 \
  --this_run_num 0 \
  --cross_dataset_test 1 \
  --force_LODO 1 \
  --exp_name_rigid LODO \
  --pretrained 0 \
  --tuned_config ./configs/best_configs_augmented/LODO/POTR_BMCLABS_LODO/0/best_params.json
```

### 예시: within (T-SDU-PD, POTR)

```bash
cd data/datasets/CARE-PD-code
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled WANDB_SILENT=true \
./../../../web/.venv312/bin/python eval_only.py \
  --backbone potr \
  --config T-SDU-PD.json \
  --hypertune 0 \
  --this_run_num 0 \
  --num_folds -1 \
  --pretrained 0 \
  --tuned_config ./configs/best_configs_augmented/Hypertune/POTR_TRI_PD/0/best_params.json
```

## 6) 결과 파일 위치

공식 코드 출력 기본 경로:

- `data/datasets/CARE-PD-code/experiment_outs/**`

리포트/예측 결과는 각 실험 폴더의:

- `.../last_report_allfolds_just012updrs.txt`
- `.../final_results.pkl`

등에서 수집.

## 7) 실행 중 모니터링

```bash
ps -axo pid,etime,pcpu,pmem,command | rg "eval_only.py|run.py" | rg -v rg
```

## 8) 비교표 템플릿

| Protocol | Dataset | RF Macro-F1 | Official Macro-F1 | Delta (Official-RF) |
|---|---|---:|---:|---:|
| Within (LOSO) | BMCLab |  |  |  |
| Within (LOSO) | 3DGait |  |  |  |
| Within (LOSO) | PD-GaM |  |  |  |
| Within (LOSO) | T-SDU-PD |  |  |  |
| LODO | BMCLab(target) |  |  |  |
| LODO | 3DGait(target) |  |  |  |
| LODO | PD-GaM(target) |  |  |  |
| LODO | T-SDU-PD(target) |  |  |  |

## 9) 재현성 노트

- OpenMP 이슈 회피: `OMP_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE`
- Python 3.12 + chumpy 호환 이슈가 있어 공식 코드 일부 호환 패치가 필요할 수 있음
- CPU-only 환경에서는 official full sweep 시간이 매우 길어짐
