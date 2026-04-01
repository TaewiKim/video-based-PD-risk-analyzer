from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from django.conf import settings
from django.utils import timezone

from .models import RateLimitEvent


@dataclass(frozen=True)
class RateLimitRule:
    action: str
    limit: int
    window: timedelta


def subject_ip(request) -> str:
    return f"ip:{request.META.get('REMOTE_ADDR', 'anonymous')}"


def subject_user(request) -> str:
    user = getattr(request, "user", None)
    if user is not None and getattr(user, "is_authenticated", False):
        return f"user:{user.pk}"
    return subject_ip(request)


def _check(rule: RateLimitRule, subject_key: str) -> tuple[bool, int]:
    now = timezone.now()
    window_start = now - rule.window
    count = RateLimitEvent.objects.filter(
        action=rule.action,
        subject_key=subject_key,
        created_at__gte=window_start,
    ).count()
    remaining = max(rule.limit - count, 0)
    return count < rule.limit, remaining


def enforce_rate_limit(rule: RateLimitRule, subject_key: str) -> tuple[bool, int]:
    allowed, remaining = _check(rule, subject_key)
    if not allowed:
        return False, 0
    RateLimitEvent.objects.create(action=rule.action, subject_key=subject_key)
    return True, max(remaining - 1, 0)


def auth_rules() -> dict[str, RateLimitRule]:
    return {
        "register_ip": RateLimitRule(
            action="register_ip",
            limit=int(getattr(settings, "AUTH_REGISTER_PER_IP_HOUR", 5)),
            window=timedelta(hours=1),
        ),
        "register_email": RateLimitRule(
            action="register_email",
            limit=int(getattr(settings, "AUTH_REGISTER_PER_EMAIL_HOUR", 3)),
            window=timedelta(hours=1),
        ),
        "resend_email": RateLimitRule(
            action="resend_email",
            limit=int(getattr(settings, "AUTH_RESEND_PER_EMAIL_HOUR", 5)),
            window=timedelta(hours=1),
        ),
        "login_ip": RateLimitRule(
            action="login_ip",
            limit=int(getattr(settings, "AUTH_LOGIN_PER_IP_WINDOW", 10)),
            window=timedelta(minutes=int(getattr(settings, "AUTH_LOGIN_WINDOW_MINUTES", 15))),
        ),
        "login_user": RateLimitRule(
            action="login_user",
            limit=int(getattr(settings, "AUTH_LOGIN_PER_USER_WINDOW", 8)),
            window=timedelta(minutes=int(getattr(settings, "AUTH_LOGIN_WINDOW_MINUTES", 15))),
        ),
        "verify_ip": RateLimitRule(
            action="verify_ip",
            limit=int(getattr(settings, "AUTH_VERIFY_PER_IP_HOUR", 20)),
            window=timedelta(hours=1),
        ),
    }


def analysis_rules() -> dict[str, RateLimitRule]:
    return {
        "analysis_hour": RateLimitRule(
            action="analysis_hour",
            limit=int(getattr(settings, "ANALYSIS_PER_USER_HOUR", 12)),
            window=timedelta(hours=1),
        ),
        "analysis_day": RateLimitRule(
            action="analysis_day",
            limit=int(getattr(settings, "ANALYSIS_PER_USER_DAY", 40)),
            window=timedelta(days=1),
        ),
    }
