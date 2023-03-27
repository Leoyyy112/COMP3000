from django.db import models
from django.contrib.auth.models import User
# Create your models here.
from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    upload_date = models.DateTimeField(auto_now_add=True)

    owner = models.ForeignKey(User,on_delete=models.CASCADE)

    def __str__(self):
        return self.image.name
