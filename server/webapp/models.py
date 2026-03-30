from django.db import models


class PersonUsage(models.Model):
    person_id = models.CharField(max_length=120, unique=True)
    used_count = models.PositiveIntegerField()
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)


class UsageEvent(models.Model):
    person = models.ForeignKey(PersonUsage, on_delete=models.CASCADE, related_name="events")
    created_at = models.DateTimeField(auto_now_add=True)


class AnalysisResult(models.Model):
    RESULT_TYPE_CHOICES = [
        ("gait", "Gait"),
        ("symptoms", "Symptoms"),
    ]

    result_filename = models.CharField(max_length=255, unique=True)
    result_type = models.CharField(max_length=32, choices=RESULT_TYPE_CHOICES)
    video_filename = models.CharField(max_length=255, blank=True, default="")
    payload = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
