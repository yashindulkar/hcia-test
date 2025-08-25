from django.db import models

class Feedback(models.Model):
    movie_title = models.CharField(max_length=255)
    helpfulness = models.CharField(max_length=50)
    comments = models.TextField(blank=True)
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.movie_title} - {self.helpfulness}"