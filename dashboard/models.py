from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class UploadedFile(models.Model):
    title = models.CharField(max_length=200,null=True,blank=True)
    file = models.FileField(upload_to='uploads/')
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
    
class BillionairesWishlist(models.Model):
    name1 = models.CharField(max_length=200,null=True,blank=True)
    name2 = models.CharField(max_length=200,null=True,blank=True)
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.name1} vs {self.name2}'