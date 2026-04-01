# API Reference

Base paths:

- UI-friendly aliases: `/upload`, `/analyze`, `/analyze-symptoms`, ...
- Explicit API paths: `/api/upload`, `/api/analyze`, `/api/analyze-symptoms`, ...

The bundled frontend currently uses the non-`/api` aliases in some flows, but new integrations should prefer the explicit `/api/*` routes.

Access model:

- the main UI requires an authenticated Django session
- protected API routes return `401` JSON when unauthenticated
- analysis artifacts are session-scoped by default
- uploaded videos, saved results, and registered face photos are visible only to the browser session that created or unlocked them
- analysis requests are accepted only for videos owned by the current browser session
- async job status is visible only to the browser session that created that job
- requests outside the allowed session return `403`

## Authentication

### `GET /login`

Renders the login page for Django session authentication.

### `POST /login`

Form login endpoint.

Form fields:

- `username`
- `password`
- `next` (optional)

Current behavior:

- inactive users must verify email before login
- repeated login attempts are rate-limited by IP and account key

### `GET /register`

Renders the email-verification sign-up page.

### `POST /register`

Creates an inactive user account and sends a verification email.

Form fields:

- `username`
- `email`
- `password`
- `password_confirm`

Current behavior:

- new accounts start with `is_active=false`
- the response renders a success message after sending the verification email
- repeated sign-up attempts are rate-limited by IP and email address

### `GET /verify-email?token=...`

Consumes an email verification token, activates the account, and starts a logged-in session.

### `GET /resend-verification`

Renders the verification-email resend form.

### `POST /resend-verification`

Reissues a verification email for an inactive account email address.

Current behavior:

- always returns a generic success message for unknown/inactive email privacy
- repeated resend attempts are rate-limited per email address

### `POST /logout`

Clears the current Django session and redirects to `/login`.

### `GET /api/auth/session`

Returns the current authenticated user session.

Response shape:

```json
{
  "authenticated": true,
  "username": "alice",
  "is_staff": false,
  "is_superuser": false
}
```

## Rate Limits

The product uses database-backed rate limiting instead of a cache database.

Current protected actions:

- login attempts
- sign-up attempts
- email verification requests
- analysis job creation and sync analysis execution

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

Session behavior:

- only returns results linked to videos accessible in the current browser session

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

Session behavior:

- returns `403` when the current browser session is not allowed to access the linked result

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
- the current browser session is granted access to the normalized upload
- the response includes `input_processing` so the UI can disclose the resize/transcode step

Response shape:

```json
{
  "success": true,
  "filename": "uuid.mp4",
  "video_url": "/videos/uuid.mp4",
  "input_processing": {
    "normalized_for_analysis": true,
    "source_extension": "webm",
    "output_extension": "mp4",
    "video_codec": "h264",
    "audio_removed": true,
    "max_side_px": 320
  }
}
```

Possible error responses:

- `400` when no file is provided
- `400` when the extension is not allowed
- `500` when normalization fails

### `GET /videos/<filename>`

Streams an uploaded video file.

Session behavior:

- returns `403` unless the current browser session uploaded or unlocked that video

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

Session behavior:

- only returns users registered in the current browser session

### `GET /api/users/<user_id>/photo`

Returns stored face image for a user.

Session behavior:

- returns `403` unless the current browser session registered that user

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
- includes `input_processing` so the UI can disclose the analysis copy transformation
- returns `403` when the video was not uploaded in the current browser session

### `POST /api/analyze-async`

Starts a gait analysis job and returns immediately.

Session behavior:

- returns `403` when the video was not uploaded in the current browser session
- grants the current browser session access to the created job

Response shape:

```json
{
  "job_id": "job123",
  "status": "queued",
  "job_type": "gait",
  "video_filename": "uuid.mp4",
  "usage": {
    "client_id": "ip:127.0.0.1"
  }
}
```

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
- includes `input_processing` so the UI can disclose the analysis copy transformation
- returns `403` when the video was not uploaded in the current browser session

### `POST /api/analyze-symptoms-async`

Starts a full symptom-plus-gait analysis job and returns immediately.

Session behavior:

- returns `403` when the video was not uploaded in the current browser session
- grants the current browser session access to the created job

### `GET /api/jobs/<job_id>`

Returns job status for a session-owned analysis job.

Typical statuses:

- `queued`
- `running`
- `succeeded`
- `failed`

Current behavior:

- when a job succeeds and includes `result_filename`, the current browser session is granted access to that saved result

## Reference

### `GET /api/reference-data`

Returns literature/reference benchmark values and runtime model metadata used by the web UI.
