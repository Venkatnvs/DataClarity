from django.shortcuts import render,redirect,HttpResponse
from .models import UploadedFile,BillionairesWishlist
from .forms import UploadFileForm, UserCreateForm
import requests
from django.http import JsonResponse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta
from django.core.cache import cache
from django.views import View
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
import json
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import Count
from django.db.models.functions import TruncMonth
import xlwt
import csv
from django.db.models import F,Value, CharField
from django.db.models.functions import Concat
from django.contrib import messages
import os
from django.conf import settings

User = get_user_model()

@login_required()
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save(commit=True, user=request.user)
            return redirect('dashboard-file-analysis', file_id=uploaded_file.id)
    else:
        form = UploadFileForm()
    return render(request, 'main/upload_file.html', {'form': form})



@login_required()
def all_user_files(request):
    user = request.user
    items = UploadedFile.objects.filter(user=user)
    return render(request, 'main/all_files.html', {'items': items})

def chart_view():
    country_data = User.objects.values('gender').annotate(count=Count('id'))
    monthly_data = User.objects.annotate(month=TruncMonth('date_joined')).values('month').annotate(count=Count('id'))

    country_data_list = list(country_data)
    monthly_data_list = list(monthly_data)

    return country_data_list,monthly_data_list

@login_required()
def main(request):
    users = User.objects.all().count()
    analysis = UploadedFile.objects.filter(user=request.user).count()
    a,b = chart_view()
    context = {
        'user_count':users,
        'analysis_count':analysis,
        'contact_msg_count':0,
        "graphs_1":json.dumps(a,cls=DjangoJSONEncoder),
        "graphs_2":json.dumps(b,cls=DjangoJSONEncoder),
    }
    return render(request,'main/index.html',context)

@login_required()
def BillionairesView(request):
    return render(request,'main/billionair_data.html')

class BillionairesSearch(LoginRequiredMixin,View):
    path = os.path.join(settings.BASE_DIR,'dataset','billionair_data.json')
    def get(self,request):
        search_query = request.GET.get('search', '').lower()
        get_type = request.GET.get('get', '').lower()
        data = cache.get('billionaires_data',None)
        if data is None:
            with ThreadPoolExecutor() as executor:
                data = executor.submit(self.fetch_billionaires).result()
            cache.set('billionaires_data', data, timeout=1800)

        if data:
            final_data = []
            for i in data:
                if search_query in i['person']['name'].lower():
                    if get_type=='full':
                        final_data.append(i)
                    else:
                        square_image = i['person'].get('squareImage', 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/450px-No_image_available.svg.png')
                        billionaire_info = {
                            'squareImage': square_image,
                            'name': i['person']['name'],
                            'NetWorth': i['finalWorth'],
                            'age': self.calculate_age(i['birthDate']),
                            'gender': i['gender'],
                            'Residence': f"{i.get('city', '')}, {i.get('state', '')}",
                            'Source': i.get('source', ''),
                            'rank': i['rank']
                        }
                        final_data.append(billionaire_info)
            return JsonResponse(final_data, safe=False)
        else:
            return JsonResponse({'error': 'Failed to fetch data from the API'}, status=500)
        
    def fetch_billionaires(self):
        with open(self.path,'r') as f:
            data = f.read()
        if data:
            try:
                json_data = json.loads(data)
                return json_data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return None
        else:
            return None
        
    def calculate_age(self,birth_date):
        birth_datetime = datetime.utcfromtimestamp(0) + timedelta(seconds=birth_date / 1000)
        today = datetime.utcnow()
        age = today.year - birth_datetime.year - ((today.month, today.day) < (birth_datetime.month, birth_datetime.day))
        return age

@login_required()
def BilWishList(request):
    data = BillionairesWishlist.objects.filter(user=request.user)
    return render(request,'main/bil_wishlist.html',{'data':data})

@login_required()
def AllUsersView(request):
    if not request.user.is_superuser:
        return HttpResponse("You Don't have Access to this page.")
    data = User.objects.all()
    return render(request,'main/users/all_users.html',{'data':data})

@login_required()
def BusinessGrowthcls(request):
    return render(request,'main/business_growth_c.html')

@login_required()
def QuizPage(request):
    return render(request,'main/quiz_page.html')

class UserAddView(LoginRequiredMixin,View):
    def get(self,request):
        form = UserCreateForm()
        return render(request,'main/users/user_create.html',{'form':form})
    
    def post(self,request):
        form = UserCreateForm(request.POST)
        if form.is_valid():
            save_file = form.save(commit=True)
            return redirect('main-users-all')
        messages.error(request,'Some thing went Wrong') 
        return redirect('main-user-add')

# Export Data Users
@login_required()
def UserDataExportExcel(request):
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename=UserData'+str(datetime.now())+'.xls'
    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet('UserInfo')
    row_num = 0
    font_style = xlwt.XFStyle()
    font_style.font.bold = True
    columns = ['Email','Full Name','Gender','Mobile No.','Verification Status','Created at']
    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num], font_style)
    font_style =  xlwt.XFStyle()
    rows = User.objects.all().annotate(
    full_name=Concat(F('first_name'), Value(' '), F('last_name'), output_field=CharField())
        ).values_list(
            'email',
            'full_name',
            'gender',
            'mobile_no',
            'is_active',
            'date_joined')
    for row in rows:
        row_num += 1
        for col_num in range(len(row)):
            ws.write(row_num, col_num, str(row[col_num]), font_style)
    wb.save(response)
    return response

@login_required()
def UserDataExportCsv(request):
    response = HttpResponse(content_type='text/csv')
    file_name = f'UserData{str(datetime.now())}.csv'
    response['Content-Disposition'] = 'attachment; filename='+file_name
    writer = csv.writer(response)
    writer.writerow(['Email','Full Name','Gender','Mobile No.','Verification Status','Created at'])

    rows = User.objects.all().annotate(
        full_name=Concat(F('first_name'), Value(' '), F('last_name'), output_field=CharField())
        ).values_list(
            'email',
            'full_name',
            'gender',
            'mobile_no',
            'is_active',
            'date_joined')
    for row in rows:
        writer.writerow(row)
    return response