# Activity Schema

최종 업데이트: 2026-03-25

이 문서는 symptom routing activity와 gait phase activity를 하나의 공통 payload 규칙으로 정리합니다.

## 목적

기존에는 두 activity 체계가 따로 있었습니다.

- symptom analyzer: `walking`, `resting`, `task`, `standing`
- gait analyzer: `walking`, `turning`, `standing`

이 둘은 의미가 다르지만, UI와 저장 결과에서 공통된 entry point가 없어서 정합성 문제가 반복됐습니다.

이제 공통 top-level field는 `activity_schema`입니다.

## Root Schema

full-scan 결과의 root에는 다음이 들어갑니다.

```json
{
  "activity_schema": {
    "version": "2026-03-25",
    "routing": {
      "kind": "symptom_routing",
      "labels": ["walking", "resting", "task", "standing"],
      "summary": {
        "walking": 0.0,
        "resting": 0.0,
        "task": 0.0,
        "standing": 0.0,
        "total_duration": 0.0
      },
      "segments": []
    },
    "gait_phase": {
      "kind": "gait_phase",
      "labels": ["walking", "turning", "standing"],
      "summary": {
        "walking": 0.0,
        "turning": 0.0,
        "standing": 0.0,
        "walking_ratio": 0.0,
        "turning_ratio": 0.0,
        "standing_ratio": 0.0,
        "total_duration": 0.0
      },
      "segments": []
    },
    "gait_source_person_id": "person_1",
    "owner_person_id": null
  }
}
```

root `activity_schema.gait_phase`는 더 이상 optional이 아닙니다.

- live `/analyze-symptoms` 응답
- `GET /results/<filename>` history 응답
- 새로 저장되는 `_symptoms.json`

이 세 경로 모두에서 root `gait_phase`가 채워져 있어야 합니다.

## Person Schema

각 `persons[*]` 안에도 `activity_schema`가 있습니다.

- `routing`은 항상 존재
- `gait_phase`는 해당 person이 `gait_source_person_id`와 같을 때만 존재

즉 multi-person 결과에서 gait phase는 source person에게만 붙습니다.

## Legacy Fields

기존 필드는 호환 때문에 유지합니다.

- root `activity_summary`
- person `activity_breakdown`
- person `activity_segments`
- `gait_analysis.activity_timeline`

하지만 새 UI와 새 코드 경로는 `activity_schema`를 우선 사용해야 합니다.

## History Compatibility

예전 `_symptoms.json` 파일은 `activity_schema`가 없을 수 있습니다.

현재 `GET /results/<filename>` load path에서는:

- legacy payload를 읽고
- `activity_schema`를 동적으로 보정해서 반환합니다

즉 저장 파일이 구버전이어도 UI는 새 schema를 우선 사용할 수 있어야 합니다.

추가 보정 규칙:

- root `activity_schema.gait_phase`가 비어 있으면 `gait_analysis.activity_timeline`에서 복원
- source person의 `persons[*].activity_schema.gait_phase`가 비어 있으면 동일하게 복원
- `persons[*].skeleton_track.frames`가 없고 `keypoints`만 있으면 `frames` alias를 동적으로 추가
