Run Django backend + React frontend:

1) Install dependency

   uv sync

2) Run migrations

   python demo_django/manage.py migrate

3) Start server

   python demo_django/manage.py runserver

Open http://127.0.0.1:8000

API:

- GET /api/status
- POST /api/upload (multipart form-data, key: video)
- POST /api/analyze with JSON {"filename":"...","identify_user":false}
- POST /api/analyze-symptoms with JSON {"filename":"..."}
- GET /api/reference-data
- POST /api/register-user with JSON {"name":"...","image":"data:image/...base64,..."}
- GET /api/users
- GET /api/users/<user_id>/photo
- GET /videos/<filename>

Anonymous client usage is limited to 10 analysis calls.
