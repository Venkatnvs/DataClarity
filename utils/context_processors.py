from django.conf import settings

def Appdata(request):
    context = {
        'site_name':settings.SITE_NAME,
    }
    return context