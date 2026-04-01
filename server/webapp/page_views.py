from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.views.decorators.http import require_GET


@login_required(login_url="login")
def index(request):
    return render(request, "webapp/index.html")


@require_GET
def favicon(request):
    return HttpResponse(status=204)
