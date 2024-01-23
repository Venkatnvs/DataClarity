from django import forms
from .models import UploadedFile
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm

User = get_user_model()

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['file','title']

    def save(self, commit=True, user=None):
        instance = super().save(commit=False)
        if user:
            instance.user = user
        if commit:
            instance.save()
        return instance
    
class UserCreateForm(UserCreationForm):
    class Meta: 
        model = User
        fields = ['email','password','first_name','last_name','gender','mobile_no']
            