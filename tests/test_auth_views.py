from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.core import mail
from django.test import Client


@pytest.mark.django_db
def test_login_page_redirects_authenticated_user() -> None:
    user = get_user_model().objects.create_user(username="alice", password="secret123")
    client = Client()
    assert client.login(username="alice", password="secret123")

    response = client.get("/login")

    assert response.status_code == 302
    assert response.headers["Location"] == "/"


@pytest.mark.django_db
def test_login_and_logout_flow() -> None:
    get_user_model().objects.create_user(username="alice", password="secret123")
    client = Client()

    response = client.post("/login", {"username": "alice", "password": "secret123", "next": "/"})
    assert response.status_code == 302
    assert response.headers["Location"] == "/"

    session_response = client.get("/api/auth/session")
    assert session_response.status_code == 200
    assert session_response.json()["username"] == "alice"

    logout_response = client.post("/logout")
    assert logout_response.status_code == 302
    assert logout_response.headers["Location"] == "/login"

    session_after_logout = client.get("/api/auth/session")
    assert session_after_logout.status_code == 401


@pytest.mark.django_db
def test_login_without_remember_me_expires_at_browser_close() -> None:
    get_user_model().objects.create_user(username="alice", password="secret123")
    client = Client()

    response = client.post("/login", {"username": "alice", "password": "secret123", "next": "/"})

    assert response.status_code == 302
    session = client.session
    assert session.get_expire_at_browser_close() is True


@pytest.mark.django_db
def test_login_with_remember_me_uses_persistent_session(settings) -> None:
    settings.SESSION_COOKIE_AGE = 7200
    get_user_model().objects.create_user(username="alice", password="secret123")
    client = Client()

    response = client.post(
        "/login",
        {"username": "alice", "password": "secret123", "next": "/", "remember_me": "on"},
    )

    assert response.status_code == 302
    session = client.session
    assert session.get_expire_at_browser_close() is False
    assert session.get_expiry_age() == 7200


@pytest.mark.django_db
def test_register_sends_verification_email_and_keeps_user_inactive() -> None:
    client = Client()

    response = client.post(
        "/register",
        {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "secret123",
            "password_confirm": "secret123",
        },
    )

    assert response.status_code == 201
    user = get_user_model().objects.get(username="newuser")
    assert user.is_active is False
    assert len(mail.outbox) == 1
    assert "/verify-email?token=" in mail.outbox[0].body


@pytest.mark.django_db
def test_inactive_user_cannot_login_until_email_verified() -> None:
    get_user_model().objects.create_user(
        username="pending",
        email="pending@example.com",
        password="secret123",
        is_active=False,
    )
    client = Client()

    response = client.post("/login", {"username": "pending", "password": "secret123", "next": "/"})

    assert response.status_code == 200
    assert b"Verify your email before logging in" in response.content


@pytest.mark.django_db
def test_verify_email_activates_account_and_logs_user_in() -> None:
    client = Client()
    response = client.post(
        "/register",
        {
            "username": "verifyme",
            "email": "verifyme@example.com",
            "password": "secret123",
            "password_confirm": "secret123",
        },
    )
    assert response.status_code == 201

    body = mail.outbox[0].body
    token = body.split("/verify-email?token=", 1)[1].splitlines()[0].strip()

    verify_response = client.get(f"/verify-email?token={token}")

    assert verify_response.status_code == 200
    user = get_user_model().objects.get(username="verifyme")
    user.refresh_from_db()
    assert user.is_active is True
    session_response = client.get("/api/auth/session")
    assert session_response.status_code == 200
    assert session_response.json()["username"] == "verifyme"


@pytest.mark.django_db
def test_resend_verification_sends_new_email_for_inactive_user() -> None:
    get_user_model().objects.create_user(
        username="pending",
        email="pending@example.com",
        password="secret123",
        is_active=False,
    )
    client = Client()

    response = client.post("/resend-verification", {"email": "pending@example.com"})

    assert response.status_code == 200
    assert len(mail.outbox) == 1
    assert "verify-email?token=" in mail.outbox[0].body


@pytest.mark.django_db
def test_resend_verification_rate_limit_blocks_excessive_attempts(settings) -> None:
    settings.AUTH_RESEND_PER_EMAIL_HOUR = 1
    get_user_model().objects.create_user(
        username="pending",
        email="pending@example.com",
        password="secret123",
        is_active=False,
    )
    client = Client()

    first = client.post("/resend-verification", {"email": "pending@example.com"})
    second = client.post("/resend-verification", {"email": "pending@example.com"})

    assert first.status_code == 200
    assert second.status_code == 429


@pytest.mark.django_db
def test_register_rate_limit_blocks_excessive_attempts(settings) -> None:
    settings.AUTH_REGISTER_PER_IP_HOUR = 1
    client = Client(REMOTE_ADDR="10.0.0.1")

    first = client.post(
        "/register",
        {
            "username": "first",
            "email": "first@example.com",
            "password": "secret123",
            "password_confirm": "secret123",
        },
    )
    second = client.post(
        "/register",
        {
            "username": "second",
            "email": "second@example.com",
            "password": "secret123",
            "password_confirm": "secret123",
        },
    )

    assert first.status_code == 201
    assert second.status_code == 429


@pytest.mark.django_db
def test_login_rate_limit_blocks_excessive_attempts(settings) -> None:
    settings.AUTH_LOGIN_PER_IP_WINDOW = 1
    get_user_model().objects.create_user(username="alice", password="secret123")
    client = Client(REMOTE_ADDR="10.0.0.1")

    first = client.post("/login", {"username": "alice", "password": "wrong", "next": "/"})
    second = client.post("/login", {"username": "alice", "password": "wrong", "next": "/"})

    assert first.status_code == 200
    assert second.status_code == 429


@pytest.mark.django_db
def test_index_redirects_anonymous_user_to_login() -> None:
    client = Client()

    response = client.get("/")

    assert response.status_code == 302
    assert response.headers["Location"].startswith("/login?next=/")


@pytest.mark.django_db
def test_protected_api_returns_401_when_anonymous() -> None:
    client = Client()

    response = client.get("/api/status")

    assert response.status_code == 401
    assert response.json()["error"] == "Authentication required"
