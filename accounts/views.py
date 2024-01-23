from django.http import HttpResponseRedirect
from django.shortcuts import render,redirect
from django.views import View
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout,get_user_model
from .helpers import extract_first_last_name,SendEmail,token_generater
from decouple import config
from django.template.loader import get_template
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_str
from django.urls import reverse
from django.conf import settings
from validate_email import validate_email
from django.contrib.auth.tokens import PasswordResetTokenGenerator

User = get_user_model()

class Registration(View):
    def get(self, request):
        return render(request, 'accounts/register.html')

    def post(self, request):
        email = request.POST['email']
        fullname = request.POST['fullname']
        gender = request.POST['gender']
        mobile_no = request.POST['mobile_no']
        mobile_no_full = request.POST['mobile_no_full']
        password = request.POST['password']
        first_name, last_name = extract_first_last_name(fullname)
        context = {
            'FieldValues':request.POST,
        }
        if User.objects.filter(email=email).exists():
            messages.error(request,"Email already exists")
            return render(request, 'accounts/register.html',context)
        if len(password)<8:
            messages.error(request, 'Password is too short')
            return render(request, 'accounts/register.html', context)
        if len(mobile_no)>10:
            messages.error(request, 'Invalid Mobile Number')
            return render(request, 'accounts/register.html', context)
        user = User.objects.create(email=email,first_name=first_name,last_name=last_name)
        user.set_password(password)
        user.gender = gender
        user.mobile_no = mobile_no_full
        user.is_active = False
        user.is_completed = True
        user.save()
        email_tmp_path = 'emails/auth/email_verification.html'
        domain = get_current_site(request).domain
        request_main = config('REQUEST')
        uidb64 = urlsafe_base64_encode(force_bytes(user.pk))
        link = reverse('accounts-activate', kwargs={'uidb64':uidb64, 'token':token_generater.make_token(user)})
        from_mail = config('FROM_MAIL')
        activate_url = domain+link
        context_email_data = {
            'title':settings.SITE_NAME,
            'baseurl':domain+request_main,
            'activate_url':activate_url,
            'user_name':user.full_name,
            'user_email':user.email,
        }
        email_body = get_template(email_tmp_path).render(context_email_data)
        email_subject = f'Activate your account | {settings.SITE_NAME}'
        SendEmail(email_subject,email_body,from_mail,email)
        messages.info(request, 'A Verification email have been sent')
        return render(request, 'accounts/register.html',context)
    
class Verification(View):
    def get(self, request, uidb64, token):
        try:
            id = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=id)
            if not token_generater.check_token(user, token):
                messages.info(request, 'User already activated')
                return redirect('accounts-login')
            if user.is_active:
                return redirect('accounts-login')
            user.is_active=True
            user.save()
            messages.success(request, 'Acccount activated successfully')
            return redirect('accounts-login')
        except Exception as e:
            print(e)
        return redirect('accounts-login')

class Login(View):
    def get(self,request):
        context = {
            'next':request.GET.get('next','/')
        }
        return render(request,"accounts/login.html",context)
    
    def post(self,request):
        email = request.POST['email']
        password = request.POST['password']
        next = request.POST.get('next','/')
        if email and password:
            user = User.objects.filter(email=email).first()
            if user and user.check_password(password):
                if not user.is_active:
                    messages.error(request, 'Account is not activated, Please check your mail box')
                    return redirect('accounts-login')
            user = authenticate(request, email=email, password=password)
            if user:
                if user.is_active:
                    login(request, user)
                    messages.success(request, 'Welcome '+user.full_name+'! You are now logged in')
                    return HttpResponseRedirect(next)
                messages.error(request, 'Account is not activated, Please check your email')
                return redirect('accounts-login')
            messages.error(request, 'Invalid credintails,Please check')
            return redirect('accounts-login')
        messages.error(request, 'Please fill all fields')
        return redirect('accounts-login')
    
class Logout(View):
    def post(self, request):
        logout(request)
        messages.success(request, 'You have been logged out')
        return redirect('main-home')

class ForgetPassword(View):
    def get(self,request):
        return render(request,"accounts/forget_password.html")
    
    def post(self,request):
        email = request.POST['email']
        context = {
            'FieldValues':request.POST
        }
        if not validate_email(email):
            messages.error(request, 'Email is not valid')
            return render(request, 'accounts/forget_password.html', context)
        
        user = User.objects.filter(email=email)
        if user.exists():
            email_tmp_path = 'emails/auth/reset_password.html'
            uidb64 = urlsafe_base64_encode(force_bytes(user[0].pk))
            domain = get_current_site(request).domain
            link = reverse('accounts-reset-user-password', kwargs={'uidb64':uidb64, 'token':PasswordResetTokenGenerator().make_token(user[0])})
            request_main = config('REQUEST')
            from_mail = config('FROM_MAIL')
            reset_url = domain+link
            email_subject = f'Reset Password Link | {settings.SITE_NAME}'
            context_email_data = {
                    'title':settings.SITE_NAME,
                    'baseurl':domain,
                    'reset_url':reset_url,
                    'user_name':user[0].full_name,
                    'user_email':user[0].email,
            }
            email_body = get_template(email_tmp_path).render(context_email_data)
            SendEmail(email_subject,email_body,from_mail,email)
            messages.success(request, 'Email have been send to reset your password')
            return render(request, 'accounts/forget_password.html')
        
        messages.error(request, 'Email not registered, Please try again')
        return render(request, 'accounts/forget_password.html', context)
    
class SetNewPassword(View):
    def get(self,request,uidb64, token):
        context = {
            'uidb64':uidb64,
            'token':token
        }
        user_id = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=user_id)
        if not PasswordResetTokenGenerator().check_token(user, token):
            messages.info(request, 'Password reset link is invalid, Please request again')
            return render(request, 'accounts/set_new_password.html')
        return render(request,"accounts/set_new_password.html",context)
    
    def post(self, request, uidb64, token):
        context = {
            'uidb64':uidb64,
            'token':token,
        }
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 != password2:
            messages.error(request, 'Password donot match')
            return render(request, 'accounts/set_new_password.html', context)
        if len(password1)<8:
            messages.error(request, 'Password too short')
            return render(request, 'accounts/set_new_password.html', context)

        user_id = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=user_id)

        if user:
            user.set_password(password1)
            user.save()
            messages.success(request, 'Password Reset successfully, You can login with new credintials')
            return redirect('accounts-login')


        messages.error(request, 'Password Reset link Not Valid')
        return render(request, 'accounts/set_new_password.html', context)