from django.urls import path, include
from .views import *
from django.views.decorators.csrf import csrf_exempt


urlpatterns = [
    path('validate-email', csrf_exempt(EmailValidation.as_view()), name='utils-validate-email'),
]