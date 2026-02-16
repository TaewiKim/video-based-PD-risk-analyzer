# Research-Automation 통합 기술 문서

최종 업데이트: 2026-02-13  
대상 저장소: `research-automation`

이 문서는 프로젝트의 전체 범위를 한 번에 이해/운영할 수 있도록 아래를 통합 정리합니다.

- 프로젝트 개념과 목표
- 문헌/데이터셋 근거
- 아키텍처 설계
- 분석 프로세스(파이프라인)
- 분석 파라미터와 분석 모델
- 평가 방법과 현재 결과

---

## 1. 프로젝트 개념

본 프로젝트는 **비디오 기반 신경계/운동 증상 분석 자동화**를 목표로 합니다.

- 입력: 일반 영상(`.mp4`, `.mov` 등)
- 처리: 얼굴 인식 + 자세/활동 분석 + 증상/위험도 추정
- 출력: 증상별 정량 지표, 중증도(severity), 통합 리포트

핵심 도메인:

- Parkinson’s Disease(PD) 운동증상: tremor, bradykinesia, posture, FOG, gait
- 임상 척도 연계: MDS-UPDRS 기반 해석(일부 모듈)

---

## 2. 문헌/데이터 근거

### 2.1 핵심 데이터셋/논문 축

코드상 문헌 카탈로그는 `scripts/download_all_papers.py`에서 관리합니다.

- CARE-PD: 멀티사이트 PD gait 익명화 데이터셋
- 3DGait: 비디오 기반 보행 분석(인지질환 포함)
- BMCLab: PD full-body kinematics/kinetics 공개 데이터
- PD-GaM, T-SDU-PD, DNE, E-LC, KUL-DT-T, T-LTC, T-SDU 관련 논문
- 임상척도 원문: MDS-UPDRS(Goetz 2008), Hoehn & Yahr, House-Brackmann

로컬 PDF 예시:

- `data/papers/goetz_2008_mds_updrs.pdf`
- `data/papers/Video-based gait analysis for assessing Alzheimerâ__s Disease and Dementia with Lewy Bodies.pdf`
- `data/papers/pmid_36875659.pdf`

### 2.2 데이터 위치

- CARE-PD 데이터: `data/datasets/CARE-PD`
- CARE-PD 공식 코드: `data/datasets/CARE-PD-code`
- 웹 분석 결과 JSON: `web/results/*.json`

---

## 3. 아키텍처 설계

### 3.1 레이어 구조

- `src/research_automation/literature/*`: 문헌 검색/다운로드/요약
- `src/research_automation/collection/*`: 데이터/영상 수집 및 품질 확인
- `src/research_automation/pipeline/*`: CARE-PD baseline 및 official adapter
- `src/research_automation/analysis/*`: 분석 핵심 로직(얼굴/행동/위험)
- `web/*`: Flask UI/API 및 실시간 분석 통합

### 3.2 분석 파이프라인(요구 순서 반영)

순서 고정 파이프라인은 `src/research_automation/analysis/video_pipeline.py`에 구현되어 있습니다.

1. 얼굴 인식 (`face_recognition`)
2. 행동 분석(인물별) (`behavior_analysis_per_person`)
3. 위험 분석(행동별) (`risk_analysis_per_action`)

`OrderedVideoAnalysisPipeline.STAGE_ORDER`:

- `face_recognition`
- `behavior_analysis_per_person`
- `risk_analysis_per_action`

### 3.3 웹 서비스 아키텍처

- 엔트리: `web/app.py`
- 주요 API:
  - `POST /upload`
  - `POST /analyze`
  - `POST /analyze-symptoms`
  - `GET /reference-data`
- 템플릿/UI: `web/templates/index.html`

---

## 4. 분석 프로세스

## 4.1 코어(Ordered Pipeline)

파일: `src/research_automation/analysis/video_pipeline.py`

1. `FaceRecognitionAnalyzer.analyze_video()`
2. `PoseExtractor.extract_from_video()`로 포즈 공통 추출
3. `BehaviorAnalyzer.analyze_person()`
   - Walking detection
   - Tremor analysis
   - Bradykinesia analysis
4. `ActionRiskAnalyzer.analyze_person()`
   - Walking risk
   - Tremor risk
   - Bradykinesia risk
   - + FOG feature 병합

## 4.2 웹 통합 증상 분석

파일: `web/pd_symptoms_analyzer.py`, `web/app.py`

1. 멀티인물 트래킹
2. Activity segmentation
   - walking/resting/task/standing
3. 활동별 증상 분석 매핑
   - walking -> gait
   - resting -> tremor
   - task -> bradykinesia
   - standing -> posture
4. standing↔walking 전이 구간 FOG 분석
5. 인물별 통계 집계 + skeleton overlay payload 생성
6. `web/app.py`에서 gait analyzer 결과와 통합 반환

---

## 5. 분석 파라미터(주요 기본값)

## 5.1 얼굴 인식

파일: `src/research_automation/analysis/face_analysis/recognition.py`

- `detection_threshold=0.05`
- `match_threshold=0.90`
- 얼굴이 검출되었지만 등록 프로파일이 없으면 기본 `person_1` 할당

## 5.2 보행 검출

파일: `src/research_automation/analysis/behavior_analysis/walking_detection.py`

- `min_walking_duration=1.0` sec
- `speed_threshold=0.01`
- `rhythm_threshold=0.3`
- `smoothing_window=0.5` sec
- confidence 결합식:
  - `0.35*speed + 0.30*rhythm + 0.20*progression + 0.15*oscillation`
- 걷기 판정 threshold: `walking_confidence > 0.4`

## 5.3 떨림(Tremor)

파일: `src/research_automation/analysis/behavior_analysis/tremor.py`

- 주파수 탐색 범위: `3.0~12.0 Hz`
- `amplitude_threshold=0.01`
- `window_size=64`
- PD rest/postural/action frequency rules:
  - rest: `3~6Hz`
  - postural: `4~8Hz`
  - action: `>8Hz`

## 5.4 Bradykinesia

파일: `src/research_automation/analysis/behavior_analysis/bradykinesia.py`

- `pause_threshold=0.02`
- `min_pause_duration=0.1` sec
- 점수(`0~4` 유사 스케일) 구성:
  - speed decrement
  - amplitude decrement
  - hesitation
  - irregularity

## 5.5 FOG (코어 리스크 모듈)

파일: `src/research_automation/analysis/risk_analysis/fog.py`

- Freeze band: `3~8 Hz`
- Locomotion band: `0.5~3 Hz`
- `min_episode_duration=0.5` sec
- `movement_threshold=0.02`
- `window_size=1.0` sec

## 5.6 FOG Transition (웹 증상 모듈)

파일: `web/pd_symptoms_analyzer.py`

- 전이 윈도우: `transition_window=2.0` sec
- `HESITATION_VELOCITY_THRESHOLD=15.0 px/s`
- Freeze Index 기준:
  - mild `>2.0`
  - moderate `>3.0`
  - severe `>5.0`

## 5.7 웹 Gait PD indicator

파일: `web/smart_analyzer.py`

- 주요 PD threshold:
  - `walking_speed < 0.55 m/s`
  - `stride_length < 0.60 m`
  - `asymmetry < 0.05` (현재 코드 정의 기준)
- 가중치 예:
  - speed `0.35`
  - stride_length `0.35`
  - asymmetry `0.15`

---

## 6. 분석 모델

## 6.1 RF baseline (핸드크래프트)

파일: `src/research_automation/pipeline/gait_baseline.py`

- 입력: CARE-PD `.pkl`(SMPL pose/trans/fps)
- 특징: duration/speed/pose variability/symmetry/frequency/accel/jerk
- 모델: `RandomForestClassifier`
  - `n_estimators=200`
  - `max_depth=10`
  - `min_samples_leaf=5`
  - `class_weight='balanced'`

## 6.2 CARE-PD official adapter

파일: `src/research_automation/pipeline/carepd_official.py`

- official code 실행 어댑터
- `run.py` / `eval_only.py` 기반
- 환경 변수 기본 주입:
  - `WANDB_MODE=disabled`
  - `WANDB_SILENT=true`
  - `OMP_NUM_THREADS=1`
  - `KMP_DUPLICATE_LIB_OK=TRUE`

## 6.3 공식 백본(문헌 코드)

CARE-PD 공식 코드 기준 지원 백본:

- `potr`
- `motionbert`
- `mixste`
- `motionagformer`
- `poseformerv2`
- `momask`
- `motionclip`

---

## 7. 평가 방법

## 7.1 RF 평가 프로토콜

파일: `src/research_automation/pipeline/gait_baseline.py`

- Within-dataset LOSO (subject-wise `LeaveOneGroupOut`)
- LODO (Leave-One-Dataset-Out)
- 지표:
  - Macro-F1
  - Accuracy
  - Balanced Accuracy

실행:

```bash
./web/.venv312/bin/python src/research_automation/pipeline/gait_baseline.py data/datasets/CARE-PD --protocol literature --method rf
```

## 7.2 Official 평가 프로토콜

- within(LOSO): `--num_folds -1`
- LODO: `--cross_dataset_test 1 --force_LODO 1`
- tuned hyperparams JSON 사용 가능:
  - `configs/best_configs_augmented/.../best_params.json`

---

## 8. 현재 평가 결과

## 8.1 RF 최신 실행 결과 (2026-02-13)

### Within (LOSO)

| Dataset | Macro-F1 | Acc | BalAcc |
|---|---:|---:|---:|
| 3DGait | 0.504 | 0.578 | 0.507 |
| BMCLab | 0.633 | 0.661 | 0.630 |
| PD-GaM | 0.661 | 0.694 | 0.628 |
| T-SDU-PD | 0.398 | 0.399 | 0.406 |
| **Mean** | **0.549** | - | - |

### LODO

| Test Dataset | Macro-F1 | Acc | BalAcc |
|---|---:|---:|---:|
| 3DGait | 0.139 | 0.200 | 0.271 |
| BMCLab | 0.140 | 0.230 | 0.328 |
| PD-GaM | 0.463 | 0.565 | 0.576 |
| T-SDU-PD | 0.274 | 0.394 | 0.441 |
| **Mean** | **0.254** | - | - |

## 8.2 웹 실분석 예시 결과 (`parkinson_gait.mp4`)

파일: `web/results/parkinson_gait_symptoms.json`

- 비디오 길이: `~19.19s`
- gait classification: `Mild Reduction`
- gait PD risk score: `28.5%`
- biomarkers:
  - stride_cv: `1.75`
  - arm_swing_asymmetry: `0.112`
  - step_time_asymmetry: `0.191`
  - walk_ratio: `0.313`
- gait_metrics:
  - speed: `0.783 m/s`
  - stride_length: `0.842 m`
  - cadence: `111.6 spm`
  - step_count: `11`

## 8.3 Official 실행 상태

현재 문헌 코드 직접 실행(LODO, POTR, BMCLab target) 진행 중.

- 프로세스 예: `eval_only.py --backbone potr --config BMCLab.json ...`
- 상태 확인 시점: CPU 100% 근접 장시간 학습

완료 후 `experiment_outs` 리포트를 수집해 RF와 동일 표 형식으로 추가해야 합니다.

---

## 9. 리스크/제약 사항

- official 코드는 GPU 전제를 강하게 가지는 구간이 있어 CPU 재현 시 시간이 매우 김
- OpenMP/환경 이슈(특히 macOS)로 환경변수 우회가 필요
- 데이터셋 전처리 누락 시 LODO가 중간에 중단됨
- 웹 분석 모듈과 코어 분석 모듈이 병렬 발전 중이라 지표 정의가 완전히 동일하지는 않음

---

## 10. 운영 가이드(실무)

1. 데이터/전처리 먼저 검증
2. RF baseline으로 빠른 smoke benchmark 확보
3. official 실험은 장시간 batch 실행 + 중간 로그 저장
4. 결과표는 반드시 protocol별(LOSO/LODO)로 분리

관련 문서:

- 재현 가이드: `docs/CAREPD_REPRO.md`

