from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET


def index(request):
    return render(request, "webapp/index.html")


@require_GET
def favicon(request):
    return HttpResponse(status=204)
