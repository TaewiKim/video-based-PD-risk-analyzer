# API Reference

Base paths:

- UI-friendly aliases: `/upload`, `/analyze`, `/analyze-symptoms`, ...
- Explicit API paths: `/api/upload`, `/api/analyze`, `/api/analyze-symptoms`, ...

The bundled frontend currently uses the non-`/api` aliases in some flows, but new integrations should prefer the explicit `/api/*` routes.

## Health / Metadata

### `GET /api/status`

Returns per-client usage metadata.

Response shape:

```json
{
  "client_id": "ip:127.0.0.1",
  "used_count": 0,
  "remaining": null,
  "limit": null,
  "unlimited": true
}
```

### `GET /api/results`

Lists saved analysis result files.

Response item shape:

```json
[
  {
    "filename": "example_symptoms.json",
    "type": "symptoms",
    "video_filename": "example.mp4",
    "size_bytes": 12345,
    "modified_ts": 1710000000.0
  }
]
```

### `GET /api/results/<filename>`

Returns one saved result payload plus linked video filename when present.

Response shape:

```json
{
  "result_filename": "example_symptoms.json",
  "video_filename": "example.mp4",
  "data": {
    "activity_schema": {}
  }
}
```

## Media

### `POST /api/upload`

`multipart/form-data` with field `video`.

Accepted input extensions:

- `mp4`
- `avi`
- `mov`
- `webm`
- `mkv`

Current behavior:

- the uploaded source file is normalized for analysis
- the stored analysis input is returned as `.mp4`

Response shape:

```json
{
  "success": true,
  "filename": "uuid.mp4",
  "video_url": "/videos/uuid.mp4"
}
```

Possible error responses:

- `400` when no file is provided
- `400` when the extension is not allowed
- `500` when normalization fails

### `GET /videos/<filename>`

Streams an uploaded video file.

### `POST /api/register-user`

JSON body:

```json
{
  "name": "Alice",
  "image": "data:image/jpeg;base64,..."
}
```

### `GET /api/users`

Lists registered users.

### `GET /api/users/<user_id>/photo`

Returns stored face image for a user.

## Analysis

### `POST /api/analyze`

JSON body:

```json
{
  "filename": "uuid.webm",
  "identify_user": false
}
```

Key response fields:

- `video_info`
- `user`
- `walking_detection`
- `preprocessing`
- `analysis_results`
- `summary`
- `statistical_analysis`
- `ml_inference`
- `usage`

Current persistence behavior:

- saves `<video-stem>_results.json`
- returns the analysis payload plus `usage`

### `POST /api/analyze-symptoms`

JSON body:

```json
{
  "filename": "uuid.webm",
  "symptoms": null
}
```

Key response fields:

- `video_info`
- `n_persons`
- `activity_summary`
- `activity_schema`
- `persons`
- `analyzed_symptoms`
- `gait_analysis`
- `usage`

Required nested `gait_analysis` fields consumed by the current UI:

- `pose_backend`
- `video_info`
- `user`
- `walking_detection`
- `analysis_results`
- `statistical_analysis`
- `ml_inference`
- `summary`
- `activity_timeline`
- `turn_detection`
- `turn_analysis`
- `turn_methodology`

Current persistence behavior:

- saves `<video-stem>_symptoms.json`
- enriches the symptom result with gait-derived fields
- normalizes activity payloads for UI/history compatibility

## Reference

### `GET /api/reference-data`

Returns literature/reference benchmark values and runtime model metadata used by the web UI.
