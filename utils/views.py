from django.shortcuts import render
from django.views import View
import json
from django.http import JsonResponse
from django.contrib.auth import get_user_model
from validate_email import validate_email

User = get_user_model()

class EmailValidation(View):
    def post(self, request):
        data = json.loads(request.body)
        email = data['email']
        if not validate_email(email):
            return JsonResponse({'email_error':'Email is invalid'})
        if User.objects.filter(email=email).exists():
            return JsonResponse({'email_error':'Sorry email is already registered'})
        return JsonResponse({'email_valid':True})
