# Models, Formulae, and References

최종 업데이트: 2026-03-23  
대상 저장소: `video-based-PD-risk-analyzer`

이 문서는 현재 코드에 구현된 모델, 문헌 cutoff, 통계식, reference threshold를 코드 기준으로 정리합니다.  
아키텍처와 전체 프로세스는 [`docs/PROJECT_FULL_DOCUMENTATION.md`](/workspace/video-based-PD-risk-analyzer/docs/PROJECT_FULL_DOCUMENTATION.md)를 참고합니다.

---

## 1. 문서 범위

여기서 정리하는 대상은 다음과 같습니다.

- 웹 gait 분석기
- multi-symptom analyzer
- CARE-PD handcrafted baseline
- 1D CNN sequence model
- self-supervised sequence model
- 통계 요약식, threshold sweep, literature cutoff rule

주의:

- 이 문서는 "현재 코드에 구현된 식"을 우선 기록합니다.
- 웹 실분석 경로는 2026-03-23 기준으로 speed/stride heuristic 중심 판정에서 문헌 cutoff 중심 요약으로 이동했습니다.
- runtime ML probability는 문헌 cutoff와 분리해 기록합니다.

### 1.1 주요 참고 문헌

- RTMLib README, RTMPose/RTMW series whole-body inference API: https://github.com/Tau-J/rtmlib
- Lewek et al., *Gait & Posture* 2010, arm swing asymmetry in early PD: https://pubmed.ncbi.nlm.nih.gov/19945285/
- Frenkel-Toledo et al., *J NeuroEngineering Rehabil* 2005, stride time variability in PD: https://pubmed.ncbi.nlm.nih.gov/16053531/
- Yogev et al., *Experimental Brain Research* 2007, gait asymmetry in PD: https://pubmed.ncbi.nlm.nih.gov/16972073/
- Moore et al., *Gait & Posture* 2007, long-term monitoring / freeze-band locomotor-band context: https://pubmed.ncbi.nlm.nih.gov/17046261/
- Hill and Nantel, *PLOS One* 2019, arm swing amplitude and asymmetry/stability context: https://pubmed.ncbi.nlm.nih.gov/31860669/
- Cury et al., systematic review/meta-analysis 2022, arm-swing kinematics in PD: https://pubmed.ncbi.nlm.nih.gov/36088898/

---

## 2. Smart Gait Analyzer

파일: [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py)

### 2.1 입력과 출력

입력:

- shared 33-keypoint pose schema
- RTMLib `BodyWithFeet` lightweight backend 우선, MediaPipe fallback
- FPS
- optional user profile

출력:

- segment-level gait parameter
- PD biomarker
- literature-cutoff summary / optional runtime HGB probability
- 통계 요약
- optional runtime HGB inference summary

현재 기본 pose extractor 설정:

- backend default: `BodyWithFeet`
- RTMLib mode default: `lightweight`
- large-frame downscale: `POSE_MAX_INFERENCE_SIZE` default `640`
- optional frame subsampling: `POSE_SAMPLE_RATE`

주의:

- gait 분석은 whole-body 133점 전체보다 하체/발/상지 일부가 더 중요하므로, 현재 기본 경로는 whole-body보다 계산량이 작은 `BodyWithFeet`를 우선 사용합니다.
- 출력은 내부적으로 프로젝트 공용 33-keypoint schema로 재매핑됩니다.

### 2.2 구현된 gait parameter 식

#### 2.2.1 Speed

코드:

- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py#L549)

현재 구현:

```text
cadence = estimate_cadence(step_signal, fps)
speed = stride_length * cadence / 120
speed = clip(speed, 0.0, 2.0)
```

설명:

- cadence를 먼저 검출하고, `speed = step_length × step_frequency` 관계를 사용합니다.
- monocular 2D pose이므로 절대 길이 calibration 오차는 남습니다.

성격:

- literature-aligned gait relation

#### 2.2.2 Cadence

코드:

- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py#L556)

현재 구현:

```text
cadence = (speed / stride_length) * 120
cadence = clip(cadence, 80, 130)
```

해석:

- stride length와 speed의 비율로 cadence를 도출합니다.
- `120`은 stride-to-step 변환을 반영한 상수 역할을 합니다.

성격:

- 프로젝트 내부 파생식

#### 2.2.3 Statistical coefficient of variation

코드:

- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py#L1349)

식:

```text
CV(%) = std / mean * 100
```

용도:

- stride time variability 등 변동성 지표 요약

#### 2.2.4 95% confidence interval

코드:

- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py#L1352)

식:

```text
SE = std / sqrt(n)
CI95 = mean ± t_(0.975, n-1) * SE
```

용도:

- segment-level parameter aggregation

#### 2.2.5 One-sample t-test against threshold

코드:

- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py#L1487)

식:

```text
t = (mean - threshold) / (std / sqrt(n))
p = 1 - CDF_t(t; df=n-1)
```

판정:

```text
statistically_abnormal = (p < 0.05) and (mean > threshold)
```

용도:

- stride time CV
- arm swing asymmetry
- step time asymmetry

### 2.3 PD literature cutoff set

코드:

- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py#L479)

현재 웹 요약에 쓰는 cutoff:

| Indicator | Threshold | Direction | Source |
|---|---:|---|---|
| stride_time_cv | 2.6% | higher is worse | literature-informed clinical cutoff used by this project; aligned with Hausdorff/Frenkel-Toledo variability literature |
| arm_swing_asymmetry | 0.10 | higher is worse | Lewek et al. / arm-swing literature |
| step_time_asymmetry | 0.05 | higher is worse | Yogev et al. |

주의:

- runtime HGB 확률은 별도 `ml_inference` 블록에 노출됩니다.
- 이전 speed/stride heuristic rule은 웹 실분석 경로에서 제거되었습니다.

### 2.4 PD indicator summary 식

코드:

- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py)

현재 구현:

```text
abnormal_count = sum(indicator_i is abnormal)
risk_score = abnormal_count / 3
risk_level = Low / Moderate / High
```

설명:

- 세 개의 literature cutoff 중 몇 개가 비정상인지로 session-level 요약을 만듭니다.
- offline CARE-PD 모델을 runtime에 로드할 수 있으면 모델 probability도 함께 기록합니다.

percentile 처리:

```text
if runtime HGB available:
    percentile ≈ clip((1 - prob_pd) * 100, 0, 100)
else:
    percentile = 50
```

성격:

- literature cutoff summary
- optional model-based side channel
- percentile은 문헌 percentile이 아니라 UI 보조값입니다.

---

## 3. Walking Detection

파일:

- [`src/research_automation/analysis/behavior_analysis/walking_detection.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/walking_detection.py)
- 웹 전용 구현도 [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py)에 별도로 존재

### 3.1 Core walking confidence

패키지형 구현 식:

```text
walking_confidence =
    0.35 * speed_score +
    0.30 * rhythm_score +
    0.20 * progression_score +
    0.15 * oscillation_score
```

코드:

- [`src/research_automation/analysis/behavior_analysis/walking_detection.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/walking_detection.py#L112)

threshold:

```text
is_walking = walking_confidence > 0.4
```

웹 구현 주의:

- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py)의 웹 전용 detector는 translation alignment 이후 pelvis displacement가 거의 0이 되는 점을 반영해, ankle excursion / ankle separation change를 movement proxy로 사용합니다.
- 현재 웹 기본 threshold는 `0.30`입니다.

### 3.2 Rhythm score

요약:

- left/right ankle x-position 차이의 zero crossing 수를 사용
- walking의 alternating pattern을 근사

핵심 구현:

```text
expected_crossings ≈ len(window) / (fps / 2)
rhythm_score = min(1.0, zero_crossings / (expected_crossings * 2))
```

코드:

- [`src/research_automation/analysis/behavior_analysis/walking_detection.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/walking_detection.py#L170)

---

## 4. Tremor Analyzer

파일:

- [`src/research_automation/analysis/behavior_analysis/tremor.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/tremor.py)
- [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py)

### 4.1 Frequency range

패키지형 tremor analyzer 기본값:

```text
min_frequency = 3.0 Hz
max_frequency = 12.0 Hz
amplitude_threshold = 0.01
```

코드:

- [`src/research_automation/analysis/behavior_analysis/tremor.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/tremor.py#L79)

### 4.2 Dominant frequency estimation

요약:

- band-pass filtering
- Welch PSD
- tremor band 내 최고 power frequency를 dominant frequency로 사용

코드:

- [`src/research_automation/analysis/behavior_analysis/tremor.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/tremor.py#L178)

### 4.3 Tremor type classification

규칙:

```text
3.0-6.0 Hz -> rest tremor
4.0-8.0 Hz -> postural tremor
>8.0 Hz -> action tremor
```

코드:

- [`src/research_automation/analysis/behavior_analysis/tremor.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/tremor.py#L360)

### 4.4 Left-right asymmetry

요약식:

```text
asymmetry = |left_amp - right_amp| / (left_amp + right_amp)
```

코드:

- [`src/research_automation/analysis/behavior_analysis/tremor.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/tremor.py#L338)

---

## 5. Bradykinesia Analyzer

파일:

- [`src/research_automation/analysis/behavior_analysis/bradykinesia.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/bradykinesia.py)
- [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py)

### 5.1 기본 파라미터

```text
pause_threshold = 0.02
min_pause_duration = 0.1 sec
```

코드:

- [`src/research_automation/analysis/behavior_analysis/bradykinesia.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/bradykinesia.py#L69)

### 5.2 Speed decrement

식:

```text
speed_decrement = (first_half_speed - second_half_speed) / (first_half_speed + 1e-6)
```

코드:

- [`src/research_automation/analysis/behavior_analysis/bradykinesia.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/bradykinesia.py#L171)

### 5.3 Amplitude decrement

식:

```text
amplitude_decrement = (first_half_amp - second_half_amp) / (first_half_amp + 1e-6)
```

코드:

- [`src/research_automation/analysis/behavior_analysis/bradykinesia.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/bradykinesia.py#L184)

### 5.4 Hesitation ratio

식:

```text
hesitation_ratio = mean(speed < pause_threshold)
```

코드:

- [`src/research_automation/analysis/behavior_analysis/bradykinesia.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/bradykinesia.py#L188)

### 5.5 Bradykinesia score

코드:

- [`src/research_automation/analysis/behavior_analysis/bradykinesia.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/behavior_analysis/bradykinesia.py#L398)

식:

```text
score =
    clip(speed_decrement, 0, 1) +
    clip(amplitude_decrement, 0, 1) +
    clip(hesitation_ratio, 0, 1) +
    (1 - clip(regularity, 0, 1))
```

범위:

- 대략 `0-4`의 UPDRS-like score

---

## 6. FOG Analyzer

파일:

- [`src/research_automation/analysis/risk_analysis/fog.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/risk_analysis/fog.py)
- [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py)

### 6.1 Core FOG bands

패키지형 구현:

```text
freeze band = 3-8 Hz
locomotion band = 0.5-3 Hz
movement_threshold = 0.02
window_size = 1.0 sec
```

코드:

- [`src/research_automation/analysis/risk_analysis/fog.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/risk_analysis/fog.py#L102)

### 6.2 Freezing Index

식:

```text
freeze_power = power in 3-8 Hz
loco_power = power in 0.5-3 Hz
FI = freeze_power / (freeze_power + loco_power)
```

코드:

- [`src/research_automation/analysis/risk_analysis/fog.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/risk_analysis/fog.py#L241)

주의:

- 문헌에서 자주 쓰이는 `freeze / locomotion` ratio와는 약간 다른 정규화형 구현입니다.
- 반면 웹 symptom analyzer 주석은 Moore 계열 문헌 정의를 직접 참조합니다.

### 6.3 Episode severity and confidence

severity:

```text
severity = 1 - min(1, mean_movement / movement_threshold)
```

confidence:

```text
confidence = mean(freezing_index over aligned windows)
```

코드:

- [`src/research_automation/analysis/risk_analysis/fog.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/analysis/risk_analysis/fog.py#L312)

### 6.4 Web FOG transition logic

웹 통합 경로에서는 standing-walking 경계도 별도 사용합니다.

코드:

- [`server/webapp/views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/views.py#L96)

규칙:

- standing -> walking = initiation transition
- walking -> standing = termination transition
- 전후 standing duration이 `>= 1.0 sec`일 때 transition 생성

---

## 7. CARE-PD Handcrafted Baseline

파일:

- [`src/research_automation/pipeline/gait_baseline.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_baseline.py)

### 7.1 Domain normalization

#### 7.1.1 Temporal resampling

식:

```text
resample all frame-wise arrays from src_fps -> target_fps (default 30.0)
```

구현:

- linear interpolation

코드:

- [`src/research_automation/pipeline/gait_baseline.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_baseline.py#L103)

#### 7.1.2 Translation canonicalization

식 요약:

```text
1. subtract first-frame origin
2. compute principal motion direction by SVD/PCA on horizontal displacement
3. rotate horizontal plane so forward axis becomes +X
```

코드:

- [`src/research_automation/pipeline/gait_baseline.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_baseline.py#L52)

### 7.2 Handcrafted features

추출되는 feature 그룹:

- temporal: `duration`, `n_frames`
- translation/velocity: mean/std/max/min speed, forward/lateral/vertical velocity
- pose variability: per-joint std/range
- symmetry: hip/knee/ankle asymmetry
- frequency: gait frequency, gait regularity
- acceleration: mean/std/max
- jerk: mean/std

대표 수식:

```text
velocity_t = diff(trans_t) * fps
speed_t = ||velocity_t||
accel_t = diff(velocity_t) * fps
jerk_t = diff(accel_t) * fps
```

frequency feature:

```text
gait_frequency = argmax FFT magnitude in 0.3-3.0 Hz
gait_regularity = max_fft_mag / mean_fft_mag
```

코드:

- [`src/research_automation/pipeline/gait_baseline.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_baseline.py#L191)

### 7.3 Binary threshold optimization

코드:

- [`src/research_automation/pipeline/gait_baseline.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_baseline.py#L399)

규칙:

```text
threshold candidates = 0.05, 0.055, ..., 0.95
objective 1 = maximize accuracy
objective 2 = tie-break by macro-F1
```

### 7.4 Classifiers and hyperparameters

코드:

- [`src/research_automation/pipeline/gait_baseline.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_baseline.py#L453)

#### RandomForest

```text
n_estimators = 200
max_depth = 10
min_samples_leaf = 5
class_weight = balanced
```

#### GradientBoosting

```text
n_estimators = 200
learning_rate = 0.05
max_depth = 3
```

#### HistGradientBoosting

```text
learning_rate = 0.05
max_iter = 300
max_depth = 6
```

#### ExtraTrees

```text
n_estimators = 400
min_samples_leaf = 2
class_weight = balanced
```

#### SVC

```text
C = 2.0
kernel = rbf
gamma = scale
class_weight = balanced
probability = True
```

#### LogisticRegression

```text
C = 1.0
max_iter = 1000
class_weight = balanced
```

### 7.5 Evaluation protocol

코드상 지원:

- `StratifiedKFold`
- 별도 재현 문서에서는 `LOSO`, `LODO` 비교도 사용

지표:

- Accuracy
- Balanced Accuracy
- Macro-F1
- ROC-AUC (binary)

---

## 8. Sequence Model

파일:

- [`src/research_automation/pipeline/gait_sequence_model.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_sequence_model.py)

### 8.1 Input sequence construction

한 walk에 대해 아래를 concat 합니다.

```text
[selected pose joints, translation, velocity, speed]
```

구체적으로:

- 10개 key gait joints x 3축 = 30 channel
- translation = 3 channel
- velocity = 3 channel
- speed = 1 channel

총 channel 수는 일반적으로 37입니다.

fixed length 처리:

```text
temporal linear interpolation -> seq_len (default 128)
```

### 8.2 Temporal CNN architecture

구조:

```text
Conv1d(64, k=5) -> BN -> ReLU -> MaxPool
Conv1d(128, k=5) -> BN -> ReLU -> MaxPool
Conv1d(128, k=3) -> BN -> ReLU -> AdaptiveAvgPool
Dropout(0.25) -> Linear(n_classes)
```

코드:

- [`src/research_automation/pipeline/gait_sequence_model.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_sequence_model.py#L34)

### 8.3 Training config

기본값:

```text
seq_len = 128
target_fps = 30
n_splits = 5
epochs = 18
batch_size = 64
lr = 1e-3
weight_decay = 1e-4
patience = 8
```

optimizer:

```text
AdamW
```

loss:

```text
CrossEntropyLoss(weight=class_weight)
```

scheduler:

```text
ReduceLROnPlateau(factor=0.5, patience=2)
```

### 8.4 Threshold tuning

binary task에서는 baseline과 동일하게 `0.05-0.95` sweep을 사용합니다.

코드:

- [`src/research_automation/pipeline/gait_sequence_model.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_sequence_model.py#L139)

---

## 9. Self-Supervised Sequence Model

파일:

- [`src/research_automation/pipeline/gait_sequence_ssl.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_sequence_ssl.py)

### 9.1 SSL pretext task

정의:

- label `0`: original sequence
- label `1`: chunk-shuffled sequence

구현:

```text
n_chunks = 4
shuffle chunk order
binary classification of original vs permuted
```

코드:

- [`src/research_automation/pipeline/gait_sequence_ssl.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/gait_sequence_ssl.py#L103)

### 9.2 Encoder architecture

SSL encoder와 supervised classifier encoder는 동일한 1D CNN backbone을 사용합니다.

구조:

```text
Conv1d -> BN -> ReLU -> MaxPool
Conv1d -> BN -> ReLU -> MaxPool
Conv1d -> BN -> ReLU -> AdaptiveAvgPool
```

fine-tune 시에는 pretrained encoder weight를 classifier로 복사합니다.

### 9.3 Training config

기본값:

```text
ssl_epochs = 12
finetune_epochs = 32
batch_size = 64
lr_ssl = 1e-3
lr_finetune = 8e-4
weight_decay = 1e-4
```

---

## 10. Multi-Symptom Analyzer Thresholds and Literature Hooks

파일:

- [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py)

### 10.1 Blink rate

주석상 normal 범위:

```text
15-24 blinks/min
```

reference in code comments:

- Karson et al., 1984/1983 계열

### 10.2 Posture thresholds

주석상 reference:

- Doherty et al. 2011
- Tinazzi et al. 2015

코드 주석 위치:

- [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py#L1504)

### 10.3 Freeze Index in web symptom analyzer

코드 주석:

```text
Freeze Index (FI) = power(3-8Hz) / power(0.5-3Hz)
```

reference:

- Moore et al. 2008

코드 위치:

- [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py#L1829)

### 10.4 Step asymmetry / FOG supporting references

코드 주석 references:

- Plotnik et al. 2005
- Moore et al. 2008

---

## 11. References Used in Code or UI

아래는 코드 주석, mock reference table, UI 설명에 직접 등장하는 reference들입니다.

### 11.1 Gait / PD biomarkers

- Mirelman et al. 2019
- Hausdorff et al. 2001
- Hausdorff 2005
- Lewek et al. 2010
- Yogev et al. 2007
- Winter 1991
- Morris et al. 1998
- Hollman et al. 2011

### 11.2 Tremor / bradykinesia / posture / FOG

- Jankovic 2008
- Karson 1983 / 1984
- Espay et al. 2009
- Doherty et al. 2011
- Tinazzi et al. 2015
- Moore et al. 2008
- Plotnik et al. 2005 / 2008
- Nieuwboer et al. 2001

### 11.3 Clinical scale / protocol context

- Goetz et al. 2008 (MDS-UPDRS)

---

## 12. Interpretation Notes

### 12.1 문헌값과 코드값의 관계

현재 저장소에는 세 종류의 수치가 혼재합니다.

1. 문헌 threshold
2. CARE-PD 기반 calibration
3. 모델 probability / 연구용 부가 지표

예를 들어:

- `stride_time_cv > 2.6` 같은 값은 문헌형 clinical cutoff에 가깝습니다.
- `HGB threshold = 0.535`는 프로젝트 내부 OOF sweep 결과입니다.
- `speed = stride_length * cadence / 120`는 gait relation을 쓰되, monocular calibration 오차를 포함합니다.

문서와 결과를 해석할 때 이 셋을 구분해야 합니다.

### 12.2 실서비스 해석 주의

현재 코드는 연구/데모 목적의 분석기입니다.

- 일부는 문헌 cutoff 기반 screening summary
- 일부는 통계 요약
- 일부는 데이터셋 기반 classifier

따라서 임상 판정 시스템으로 직접 사용하기보다, screening / visualization / research workflow 성격으로 보는 것이 적절합니다.
