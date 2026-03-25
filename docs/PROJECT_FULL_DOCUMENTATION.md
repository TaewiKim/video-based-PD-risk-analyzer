# Research-Automation 통합 기술 문서

최종 업데이트: 2026-03-23  
대상 저장소: `video-based-PD-risk-analyzer`

이 문서는 현재 저장소의 실제 코드 기준으로 아래를 정리합니다.

- 저장소 목적과 구성
- 전체 아키텍처
- 웹 분석 시스템의 데이터 흐름
- 코어 분석 파이프라인
- 출력 구조와 산출물
- 실행/운영 시 주의사항

모델 정의, 임계값, 수식, 레퍼런스는 별도 문서 [`docs/MODELS_AND_FORMULAS.md`](/workspace/video-based-PD-risk-analyzer/docs/MODELS_AND_FORMULAS.md)에 분리했습니다.

---

## 1. 프로젝트 개요

이 저장소는 두 개의 층으로 구성됩니다.

1. 연구 자동화 툴킷
2. 비디오 기반 Parkinson's Disease(PD) 위험/증상 분석 웹 데모

연구 자동화 툴킷은 문헌 수집, 데이터 수집, 실험 추적, 리포트 생성을 담당합니다.  
웹 데모는 업로드된 비디오를 기반으로 보행, 떨림, 서동, 자세, FOG(Freezing of Gait) 관련 지표를 계산하고 화면에 시각화합니다.

입력과 출력은 다음과 같습니다.

- 입력: 일반 비디오 파일 (`mp4`, `avi`, `mov`, `webm`, `mkv`)
- 중간표현: 얼굴/자세/손/얼굴 랜드마크, 사람별 track, activity segment, walking segment
- 출력: gait biomarker, symptom summary, person-level aggregation, 웹 시각화용 JSON

---

## 2. 저장소 구조

### 2.1 최상위 디렉터리

- `src/research_automation/`: 연구 자동화 및 코어 분석 패키지
- `server/`: Django 서비스, 내장 UI, 분석 서비스 모듈, 런타임 데이터
- `docs/`: 재현성/프로젝트 문서
- `config/`: 설정 파일
- `tests/`: 패키지 단위 테스트

### 2.2 `src/research_automation/` 내부 역할

- `analysis/`: 얼굴 인식, 행동 분석, risk analysis, ordered video pipeline
- `pipeline/`: CARE-PD baseline, sequence model, official adapter, extractor
- `collection/`: dataset, youtube, quality, questionnaire, ecg
- `literature/`: 논문 검색/다운로드/요약
- `report/`, `submission/`, `experiment/`: 리포트/제출/실험 추적

### 2.3 `server/` 내부 역할

- [`server/webapp/views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/views.py): Django API 엔드포인트와 웹 오케스트레이션
- [`server/webapp/templates/webapp/index.html`](/workspace/video-based-PD-risk-analyzer/server/webapp/templates/webapp/index.html): 단일 페이지 UI
- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py): gait 분석, walking detection, face registration/identification, statistical summary
- [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py): multi-person PD symptom 분석기
- `server/runtime/data/`: 등록 사용자/모델/프로필 저장
- `server/runtime/uploads/`: 업로드 비디오 저장
- `server/runtime/results/`: JSON 결과 저장

---

## 3. 전체 아키텍처

### 3.1 시스템 관점

전체 시스템은 아래 세 레이어로 나눌 수 있습니다.

1. Interface Layer
2. Analysis Orchestration Layer
3. Domain Analysis / ML Layer

### 3.2 Interface Layer

웹 인터페이스는 Django + 템플릿 기반입니다.

- 엔트리포인트: [`server/webapp/views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/views.py)
- 주요 엔드포인트:
  - `GET /`
  - `POST /upload`
  - `POST /analyze`
  - `POST /analyze-symptoms`
  - `POST /register-user`
  - `GET /users`
  - `GET /users/<user_id>/photo`
  - `GET /videos/<filename>`
  - `GET /reference-data`

프런트엔드는 [`server/webapp/templates/webapp/index.html`](/workspace/video-based-PD-risk-analyzer/server/webapp/templates/webapp/index.html) 안에서 업로드, 결과 렌더링, person tab, timeline, chart, statistics table을 모두 처리합니다.

### 3.3 Analysis Orchestration Layer

웹 분석은 두 개의 분석기를 조합합니다.

1. `SmartGaitAnalyzer`
2. `PDSymptomsAnalyzer`

`/analyze`는 gait 중심 결과를 반환하고, `/analyze-symptoms`는 multi-person symptom 결과를 반환합니다.  
`/analyze-symptoms` 내부에서는 symptom 분석 결과에 gait 분석 결과를 덧붙여 최종 응답을 풍부하게 만듭니다.

### 3.4 Domain Analysis / ML Layer

도메인 분석은 다음 모듈들로 구성됩니다.

- face registration / identification
- walking detection
- gait parameter estimation
- literature-cutoff gait biomarker summary
- segment-level statistical aggregation
- multi-person tracking
- activity segmentation
- symptom-specific analyzers
- CARE-PD 재현용 분류기 및 sequence model

---

## 4. 핵심 실행 경로

### 4.1 웹 실분석 경로

실분석 서버의 메인 흐름은 다음과 같습니다.

1. `POST /upload`
2. 업로드 파일 저장
3. `POST /analyze` 또는 `POST /analyze-symptoms`
4. 분석 결과 JSON 생성
5. `server/runtime/results/*.json` 저장
6. 프런트가 JSON을 받아 카드/차트/타임라인을 렌더링

### 4.2 `/analyze` 경로

[`server/webapp/views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/views.py)의 `/analyze`는 아래 순서입니다.

1. 파일 존재 확인
2. `analyzer.analyze_video(...)` 호출
3. 결과를 `*_results.json`으로 저장
4. JSON 응답 반환

이 경로는 gait 중심 결과를 반환합니다.

- `video_info`
- `user`
- `walking_detection`
- `preprocessing`
- `analysis_results`
- `summary`
- `statistical_analysis`
- `ml_inference`

### 4.3 `/analyze-symptoms` 경로

[`server/webapp/views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/views.py)의 `/analyze-symptoms`는 아래 순서입니다.

1. `pd_symptoms_analyzer.analyze_video(...)`
2. 같은 비디오에 대해 `analyzer.analyze_video(...)` 재실행
3. gait 결과에서 walking segment 추출
4. standing-walking 경계로부터 FOG transition 유도
5. symptom 결과에 `gait_analysis` 블록 병합
6. `persons[*].symptoms.fog` 내부 transition 정보 보강
7. 결과를 `*_symptoms.json`으로 저장
8. JSON 응답 반환

즉, symptom 분석 응답은 multi-person symptom 결과를 기본으로 하고, 그 위에 single-video gait summary를 결합한 형태입니다.

---

## 5. 웹 분석 프로세스 상세

### 5.1 Smart Gait Analyzer

[`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py)는 아래 역할을 담당합니다.

1. 비디오 읽기
2. 공용 pose extractor 기반 keypoint 추출
   - 기본 경로는 RTMW whole-body ONNX 추론
   - 필요 시 MediaPipe fallback 가능
3. 얼굴 인식 기반 사용자 식별
4. 걷기 구간 탐지
5. 걷기 구간별 gait parameter 추정
6. 문헌 cutoff 기반 gait indicator 계산
7. segment-level summary / statistical summary 계산
8. optional runtime ML inference 요약

현재 `/analyze`의 rule summary는 예전 speed/stride heuristic 대신 아래 biomarker를 기준으로 구성됩니다.

- stride time CV
- arm swing asymmetry
- step time asymmetry

runtime HGB 모델은 별도 확률값으로 병기되며, literature cutoff와 같은 스케일로 취급하지 않습니다.
세션 classification/risk summary는 literature cutoff abnormal count를 기준으로 유지하고, runtime HGB는 보조 신호로만 노출합니다.

#### 5.1.1 세부 구성 요소

- `FaceRecognizer`
- `WalkingDetector`
- `GaitEstimator`
- `GaitStatisticalAnalyzer`
- `SmartGaitAnalyzer`

#### 5.1.2 출력 구조

segment 단위 결과는 `GaitAnalysisResult` dataclass로 표현됩니다.

- 시간 구간
- 보행 속도
- stride length
- cadence
- step width
- asymmetry
- stability score
- PD-specific biomarker
- risk/classification
- optional HGB probability

session 단위 결과는 아래 블록으로 요약됩니다.

- `walking_detection`
- `analysis_results`
- `summary`
- `statistical_analysis`
- `ml_inference`

### 5.2 PD Symptoms Analyzer

[`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py)는 multi-person symptom 분석 경로입니다.

핵심 클래스는 다음과 같습니다.

- `MultiPersonTracker`
- `UnifiedActivityDetector`
- `TremorSegmentDetector`
- `BradykinesiaSegmentDetector`
- `PostureSegmentDetector`
- `FOGSegmentDetector`
- `TremorAnalyzer`
- `BradykinesiaAnalyzer`
- `PostureAnalyzer`
- `FOGTransitionDetector`
- `FOGTransitionAnalyzer`
- `SymptomStatisticalAggregator`
- `GaitAnalyzer`
- `PDSymptomsAnalyzer`

#### 5.2.1 단계별 흐름

1. 비디오 메타데이터 로드
2. multi-person pose track 추출
3. 사람별 activity segmentation
4. activity type별 symptom analysis segment 생성
5. symptom analyzer 실행
6. 사람별 symptom summary 집계
7. 사람별 skeleton track payload 생성
8. 전체 `activity_summary` 및 `persons[]` 응답 조립

#### 5.2.2 activity segmentation

activity segmentation은 다음 구분을 사용합니다.

- `walking`
- `resting`
- `task`
- `standing`

그리고 activity에 따라 증상 분석을 매핑합니다.

- `walking` -> gait, FOG
- `resting` -> tremor
- `task` -> bradykinesia
- `standing` -> posture

#### 5.2.3 symptom aggregation

각 symptom은 segment-level `SymptomResult` 목록을 만들고, 이후 `PersonSymptomSummary`로 통계 요약됩니다.

사람별 응답에는 아래 정보가 포함됩니다.

- `person_id`
- `duration`
- `activity_breakdown`
- `activity_segments`
- `track_quality`
- `skeleton_track`
- `symptoms`

최상위 응답은 아래 구조입니다.

- `video_info`
- `n_persons`
- `activity_summary`
- `persons`
- `analyzed_symptoms`

---

## 6. 코어 패키지 분석 파이프라인

웹 경로와 별도로, 패키지 내부에는 ordered pipeline이 존재합니다.

파일: [`src/research_automation/analysis/video_pipeline.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/video_pipeline.py)

이 파이프라인은 아래 순서를 강제합니다.

1. `face_recognition`
2. `behavior_analysis_per_person`
3. `risk_analysis_per_action`

실행 흐름은 다음과 같습니다.

1. `FaceRecognitionAnalyzer.analyze_video()`
2. `PoseExtractor.extract_from_video()`
3. `BehaviorAnalyzer.analyze_person()`
4. `ActionRiskAnalyzer.analyze_person()`

이 경로는 `src/research_automation/analysis/*` 기반의 보다 패키지화된 분석 API이며, 웹 서비스는 별도의 `server/webapp/services/*` 구현을 사용합니다.

즉, 저장소에는 현재 두 개의 분석 축이 공존합니다.

- 패키지형 ordered pipeline
- Django 서비스용 analyzer stack

---

## 7. ML / 분석 구성요소 맵

### 7.1 CARE-PD handcrafted baseline

파일: [`src/research_automation/pipeline/gait_baseline.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_baseline.py)

역할:

- CARE-PD `.pkl` 로드
- domain normalization
- handcrafted gait feature extraction
- classical ML classifier 학습/평가

지원 classifier:

- RandomForest
- GradientBoosting
- HistGradientBoosting
- ExtraTrees
- SVC
- LogisticRegression
- optional XGBoost

### 7.2 Sequence model

파일: [`src/research_automation/pipeline/gait_sequence_model.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_sequence_model.py)

역할:

- walk당 fixed-length sequence 생성
- 1D temporal CNN 학습
- stratified CV로 OOF probability 산출
- binary task에서 threshold 최적화

### 7.3 Self-supervised + fine-tune model

파일: [`src/research_automation/pipeline/gait_sequence_ssl.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_sequence_ssl.py)

역할:

1. temporal order verification으로 encoder 사전학습
2. supervised classifier fine-tuning

### 7.4 CARE-PD official adapter

파일: [`src/research_automation/pipeline/carepd_official.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/carepd_official.py)

역할:

- official code wrapper
- experiment 실행/평가 어댑트
- CARE-PD reproducibility 지원

자세한 재현 절차는 [`docs/CAREPD_REPRO.md`](/workspace/video-based-PD-risk-analyzer/docs/CAREPD_REPRO.md)를 참고합니다.

---

## 8. 데이터 흐름

### 8.1 비디오 -> gait 결과

```text
video file
-> frame decode
-> pose landmarks
-> walking segmentation
-> per-segment gait parameter estimation
-> PD indicator scoring
-> multi-segment statistical aggregation
-> JSON response / saved result
```

### 8.2 비디오 -> multi-symptom 결과

```text
video file
-> multi-person tracking
-> pose landmark extraction / person tracking
-> activity segmentation
-> symptom-specific segment extraction
-> symptom analyzers
-> per-person statistical aggregation
-> gait integration + FOG transition enrichment
-> JSON response / saved result
```

### 8.3 CARE-PD training/evaluation 흐름

```text
CARE-PD .pkl
-> pose/trans/fps load
-> domain normalization
-> handcrafted or sequence feature build
-> cross-validation
-> metrics + optional threshold tuning
-> report / calibration metadata
```

---

## 9. 산출물과 저장 위치

### 9.1 웹 산출물

- 업로드 파일: `server/runtime/uploads/`
- gait 결과: `server/runtime/results/*_results.json`
- symptom 결과: `server/runtime/results/*_symptoms.json`
- 사용자 프로필: `server/runtime/data/profiles/`

### 9.2 모델/데이터 관련 경로

- CARE-PD dataset: `data/datasets/CARE-PD` 또는 외부 연결 경로
- CARE-PD official code: `data/datasets/CARE-PD-code`

### 9.3 문서

- 아키텍처/프로세스: [`docs/PROJECT_FULL_DOCUMENTATION.md`](/workspace/video-based-PD-risk-analyzer/docs/PROJECT_FULL_DOCUMENTATION.md)
- 모델/수식/레퍼런스: [`docs/MODELS_AND_FORMULAS.md`](/workspace/video-based-PD-risk-analyzer/docs/MODELS_AND_FORMULAS.md)
- CARE-PD 재현: [`docs/CAREPD_REPRO.md`](/workspace/video-based-PD-risk-analyzer/docs/CAREPD_REPRO.md)
- 실험 노트: [`docs/RESEARCH_NOTE_PD_RISK_2026-02-15.md`](/workspace/video-based-PD-risk-analyzer/docs/RESEARCH_NOTE_PD_RISK_2026-02-15.md)

---

## 10. 실행과 의존성 메모

### 10.1 기본 패키지

프로젝트 의존성은 [`pyproject.toml`](/workspace/video-based-PD-risk-analyzer/pyproject.toml)에 정의되어 있습니다.

현재 코드 기준으로 주로 필요한 런타임은 다음과 같습니다.

- `numpy`
- `scipy`
- `opencv-python`
- `jinja2`
- `django`
- `mediapipe` (optional extra)

### 10.2 웹 런타임 주의

웹 서비스 레이어는 Django를 사용합니다.

- [`server/webapp/views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/views.py)
- [`server/config/settings.py`](/workspace/video-based-PD-risk-analyzer/server/config/settings.py)

## 11. 현재 아키텍처 해석

현재 저장소는 완전히 단일한 production architecture라기보다, 아래가 함께 존재하는 연구/데모형 구조입니다.

1. 재사용 가능한 Python 패키지 레이어
2. CARE-PD 실험용 모델링/재현 레이어
3. Django 기반 서비스 웹/API 레이어

이 구조의 장점은 실험과 데모를 빠르게 병행할 수 있다는 점입니다.  
반면, 패키지형 분석 코드와 웹 분석 코드가 일부 중복되어 있어 장기적으로는 공통 feature extraction / scoring layer를 통합하는 리팩터링 여지가 있습니다.

---

## 12. 권장 읽기 순서

처음 읽는 경우 아래 순서를 권장합니다.

1. [`README.md`](/workspace/video-based-PD-risk-analyzer/README.md)
2. [`docs/PROJECT_FULL_DOCUMENTATION.md`](/workspace/video-based-PD-risk-analyzer/docs/PROJECT_FULL_DOCUMENTATION.md)
3. [`docs/MODELS_AND_FORMULAS.md`](/workspace/video-based-PD-risk-analyzer/docs/MODELS_AND_FORMULAS.md)
4. [`server/webapp/views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/views.py)
5. [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py)
6. [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py)
7. [`src/research_automation/pipeline/gait_baseline.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_baseline.py)
