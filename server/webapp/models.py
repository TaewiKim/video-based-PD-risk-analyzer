from django.db import models


class PersonUsage(models.Model):
    person_id = models.CharField(max_length=120, unique=True)
    used_count = models.PositiveIntegerField()
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)


class UsageEvent(models.Model):
    person = models.ForeignKey(PersonUsage, on_delete=models.CASCADE, related_name="events")
    created_at = models.DateTimeField(auto_now_add=True)
