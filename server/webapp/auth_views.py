from __future__ import annotations

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .auth_flow import register_inactive_user, send_verification_email, verify_email_token
from .auth_utils import api_login_required
from .rate_limits import auth_rules, enforce_rate_limit, subject_ip

User = get_user_model()

@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.user.is_authenticated:
        return redirect("index")

    next_url = request.GET.get("next") or request.POST.get("next") or "/"
    error = None

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        remember_me = request.POST.get("remember_me") == "on"
        allowed, _ = enforce_rate_limit(auth_rules()["login_ip"], subject_ip(request))
        if not allowed:
            error = "Too many login attempts. Please try again later."
            return render(
                request,
                "webapp/login.html",
                {"error": error, "next": next_url, "remember_me": remember_me, "username": username},
                status=429,
            )
        if username:
            allowed, _ = enforce_rate_limit(auth_rules()["login_user"], f"login:{username.lower()}")
            if not allowed:
                error = "Too many login attempts for this account. Please try again later."
                return render(
                    request,
                    "webapp/login.html",
                    {"error": error, "next": next_url, "remember_me": remember_me, "username": username},
                    status=429,
                )
        user = authenticate(request, username=username, password=password)
        if user is None:
            inactive_user = User.objects.filter(username__iexact=username).first()
            if inactive_user is not None and not inactive_user.is_active:
                error = "Verify your email before logging in"
            else:
                error = "Invalid username or password"
        else:
            login(request, user)
            if remember_me:
                request.session.set_expiry(settings.SESSION_COOKIE_AGE)
            else:
                request.session.set_expiry(0)
            return redirect(next_url)

    return render(
        request,
        "webapp/login.html",
        {
            "error": error,
            "next": next_url,
            "remember_me": request.POST.get("remember_me") == "on",
            "username": request.POST.get("username", "").strip(),
        },
    )


@require_POST
@login_required(login_url="login")
def logout_view(request):
    logout(request)
    return redirect("login")


@require_http_methods(["GET", "POST"])
def register_view(request):
    if request.user.is_authenticated:
        return redirect("index")

    context = {"error": None, "message": None}
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")
        password_confirm = request.POST.get("password_confirm", "")

        if not username or not email or not password:
            context["error"] = "Username, email, and password are required"
            return render(request, "webapp/register.html", context, status=400)
        if password != password_confirm:
            context["error"] = "Passwords do not match"
            return render(request, "webapp/register.html", context, status=400)

        allowed, _ = enforce_rate_limit(auth_rules()["register_ip"], subject_ip(request))
        if not allowed:
            context["error"] = "Too many sign-up attempts from this IP. Please try again later."
            return render(request, "webapp/register.html", context, status=429)
        allowed, _ = enforce_rate_limit(auth_rules()["register_email"], f"register:{email}")
        if not allowed:
            context["error"] = "Too many verification emails were sent to this address. Please try again later."
            return render(request, "webapp/register.html", context, status=429)

        try:
            user = register_inactive_user(username=username, email=email, password=password)
            send_verification_email(request, user, email)
        except ValueError as exc:
            context["error"] = str(exc)
            return render(request, "webapp/register.html", context, status=400)

        context["message"] = "Verification email sent. Check your inbox to activate the account."
        return render(request, "webapp/register.html", context, status=201)

    return render(request, "webapp/register.html", context)


@require_http_methods(["GET", "POST"])
def resend_verification_view(request):
    if request.user.is_authenticated:
        return redirect("index")

    context = {"error": None, "message": None}
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        if not email:
            context["error"] = "Email is required"
            return render(request, "webapp/resend_verification.html", context, status=400)

        allowed, _ = enforce_rate_limit(auth_rules()["resend_email"], f"resend:{email}")
        if not allowed:
            context["error"] = "Too many verification emails were requested for this address. Please try again later."
            return render(request, "webapp/resend_verification.html", context, status=429)

        user = User.objects.filter(email__iexact=email).first()
        if user is not None and not user.is_active:
            send_verification_email(request, user, email)
        context["message"] = "If an inactive account exists for this email, a verification email has been sent."
        return render(request, "webapp/resend_verification.html", context, status=200)

    return render(request, "webapp/resend_verification.html", context)


@require_GET
def verify_email_view(request):
    token = request.GET.get("token", "").strip()
    if not token:
        return render(
            request,
            "webapp/verify_email.html",
            {"success": False, "message": "Missing verification token"},
            status=400,
        )

    allowed, _ = enforce_rate_limit(auth_rules()["verify_ip"], subject_ip(request))
    if not allowed:
        return render(
            request,
            "webapp/verify_email.html",
            {"success": False, "message": "Too many verification attempts. Please try again later."},
            status=429,
        )

    try:
        user = verify_email_token(token)
    except ValueError as exc:
        return render(
            request,
            "webapp/verify_email.html",
            {"success": False, "message": str(exc)},
            status=400,
        )

    login(request, user)
    return render(
        request,
        "webapp/verify_email.html",
        {"success": True, "message": "Email verified. Your account is now active."},
    )


@require_GET
@api_login_required
def api_auth_session(request):
    return JsonResponse(
        {
            "authenticated": True,
            "username": request.user.get_username(),
            "is_staff": bool(request.user.is_staff),
            "is_superuser": bool(request.user.is_superuser),
        }
    )
