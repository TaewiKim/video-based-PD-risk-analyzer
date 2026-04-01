from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.mail import send_mail
from django.db import transaction
from django.urls import reverse
from django.utils import timezone

from .models import EmailVerification


User = get_user_model()


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_verification(user, email: str) -> str:
    raw_token = secrets.token_urlsafe(32)
    expires_at = timezone.now() + timedelta(hours=int(getattr(settings, "EMAIL_VERIFICATION_TTL_HOURS", 24)))
    EmailVerification.objects.filter(user=user, consumed_at__isnull=True).delete()
    EmailVerification.objects.create(
        user=user,
        email=email,
        token_hash=_hash_token(raw_token),
        expires_at=expires_at,
    )
    return raw_token


def send_verification_email(request, user, email: str) -> str:
    raw_token = create_verification(user, email)
    verify_url = request.build_absolute_uri(f"{reverse('verify_email')}?token={raw_token}")
    send_mail(
        subject="Verify your PD Gait Biomarkers account",
        message=(
            "Verify your email address to activate your account.\n\n"
            f"Verification link: {verify_url}\n\n"
            f"This link expires in {getattr(settings, 'EMAIL_VERIFICATION_TTL_HOURS', 24)} hours."
        ),
        from_email=getattr(settings, "DEFAULT_FROM_EMAIL", "no-reply@example.com"),
        recipient_list=[email],
    )
    return verify_url


@transaction.atomic
def register_inactive_user(*, username: str, email: str, password: str):
    normalized_email = email.strip().lower()
    user = User.objects.filter(email__iexact=normalized_email).first()
    if user is not None:
        if user.is_active:
            raise ValueError("An active account with this email already exists")
        user.username = username
        user.email = normalized_email
        user.set_password(password)
        user.save(update_fields=["username", "email", "password"])
        return user

    if User.objects.filter(username__iexact=username).exists():
        raise ValueError("This username is already in use")

    return User.objects.create_user(
        username=username,
        email=normalized_email,
        password=password,
        is_active=False,
    )


@transaction.atomic
def verify_email_token(raw_token: str):
    token_hash = _hash_token(raw_token)
    verification = (
        EmailVerification.objects.select_related("user")
        .filter(token_hash=token_hash, consumed_at__isnull=True)
        .first()
    )
    if verification is None:
        raise ValueError("Invalid verification token")
    if verification.expires_at <= timezone.now():
        raise ValueError("Verification token has expired")

    verification.consumed_at = timezone.now()
    verification.save(update_fields=["consumed_at"])
    user = verification.user
    user.email = verification.email
    user.is_active = True
    user.save(update_fields=["email", "is_active"])
    return user
