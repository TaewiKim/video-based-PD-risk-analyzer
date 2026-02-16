# PD Risk Research Note (2026-02-15)

## 목적
- UPDRS 라벨 기반 PD risk 분석 성능 개선
- 도메인 편차(사이트/카메라/축 불일치) 완화
- 멀티클래스(UPDRS 0/1/2/3) 및 이진 risk(UPDRS>0) 성능 비교

## 데이터
- CARE-PD (`data/datasets/CARE-PD`)
- 총 샘플: 2,953 walks
- 라벨: `UPDRS_GAIT`

## 적용한 변경

### 1) 도메인 정규화 강화
파일: `src/research_automation/pipeline/gait_baseline.py`
- 시계열 FPS 통일: `target_fps=30`
- translation 정렬:
  - 시작점 원점 정렬
  - PCA 기반 전진축 정렬(도메인별 좌표축 차이 완화)
- 속도 outlier clipping: 99 percentile

### 2) 임계값 최적화(이진)
파일: `src/research_automation/pipeline/gait_baseline.py`
- OOF probability 기반 threshold sweep(0.05~0.95)
- 기준: Accuracy 최대, 동률 시 Macro-F1 최대

### 3) 시계열 모델 추가
파일: `src/research_automation/pipeline/gait_sequence_model.py`
- Temporal CNN (Conv1D)
- class weight + early stopping + scheduler 적용

### 4) SSL + Fine-tune 모델 추가
파일: `src/research_automation/pipeline/gait_sequence_ssl.py`
- SSL pretrain: temporal order verification
- supervised fine-tune로 UPDRS 분류

### 5) 운영 기본 분류기 변경
파일: `src/research_automation/pipeline/gait_baseline.py`
- 기본 PD risk 분류기: `HistGradientBoosting`으로 변경
- `DEFAULT_PD_RISK_CLASSIFIER="histgb"`
- CLI 기본값도 `--classifier histgb`

## 실험 결과 요약

### A. Handcrafted baseline (pooled CV)
- RF (초기 기준)
  - Multiclass: Acc 0.787 / BalAcc 0.783 / Macro-F1 0.780
  - Binary: Acc 0.865 / ROC-AUC 0.943
- HistGradientBoosting (도메인 정규화 적용)
  - Multiclass: **Acc 0.813 / BalAcc 0.800 / Macro-F1 0.814**
  - Binary: Acc 0.890 / BalAcc 0.890 / ROC-AUC 0.957

### B. Binary threshold tuning
파일: `reports/pd_risk_threshold_tuning_binary_fast.json`
- HistGradientBoosting best threshold: **0.535**
- best Acc: **0.8947**

### C. Sequence models (multiclass)
- Temporal CNN (개선형): Acc 0.7948 / BalAcc 0.8269 / Macro-F1 0.7946
- SSL+Fine-tune (초기): Acc 0.7809 / BalAcc 0.7997 / Macro-F1 0.7962
- SSL+Fine-tune (튜닝: ssl 12, ft 40): **Acc 0.8029 / BalAcc 0.8109 / Macro-F1 0.8133**

참고 파일:
- `reports/pd_risk_eval_updrs_pooled.json`
- `reports/model_comparison_multiclass.json`

## 해석
- 현재 기준 최고 Accuracy:
  - Binary: HistGradientBoosting + threshold tuning (0.8947)
  - Multiclass: HistGradientBoosting handcrafted baseline (0.813)
- 시계열/SSL은 BalAcc 또는 Macro-F1 측면에서 일부 개선 가능하지만, 현 세팅에서는 Accuracy 우위는 아직 HGB가 유지.

## 다음 액션
1. 운영 경로는 `histgb` 고정 유지
2. 멀티클래스 목표 개선 시:
   - SSL augmentation 강화
   - 클래스별 focal loss/label smoothing 실험
   - subject-aware split 비교
3. 이진 risk 실서비스는 threshold 0.535 기준으로 calibration 점검 후 확정
