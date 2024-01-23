from django.contrib import admin
from .models import UploadedFile,BillionairesWishlist,BusinessData


admin.site.register(UploadedFile)
admin.site.register(BusinessData)
admin.site.register(BillionairesWishlist)
