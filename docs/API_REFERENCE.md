# API Reference

Base paths:

- UI-friendly aliases: `/upload`, `/analyze`, `/analyze-symptoms`, ...
- Explicit API paths: `/api/upload`, `/api/analyze`, `/api/analyze-symptoms`, ...

The bundled frontend uses the non-`/api` aliases, but both surfaces are available.

## Health / Metadata

### `GET /status`

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

### `GET /results`

Lists saved analysis result files.

### `GET /results/<filename>`

Returns one saved result payload plus linked video filename when present.

## Media

### `POST /upload`

`multipart/form-data` with field `video`.

Response shape:

```json
{
  "success": true,
  "filename": "uuid.webm",
  "video_url": "/videos/uuid.webm"
}
```

### `GET /videos/<filename>`

Streams an uploaded video file.

### `POST /register-user`

JSON body:

```json
{
  "name": "Alice",
  "image": "data:image/jpeg;base64,..."
}
```

### `GET /users`

Lists registered users.

### `GET /users/<user_id>/photo`

Returns stored face image for a user.

## Analysis

### `POST /analyze`

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

### `POST /analyze-symptoms`

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
- `persons`
- `analyzed_symptoms`
- `gait_analysis`
- `usage`

## Reference

### `GET /reference-data`

Returns literature/reference benchmark values and runtime model metadata used by the web UI.
