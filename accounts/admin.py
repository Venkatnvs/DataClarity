from django.contrib import admin
from .models import CustomUser
from django.contrib.auth.admin import UserAdmin

class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ['email', 'first_name','last_name', 'is_active', 'is_staff', 'is_completed']
    search_fields = ['email', 'first_name','last_name']
    list_filter = ['is_active', 'is_staff', 'is_completed']

    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal Info', {'fields': ('first_name','last_name','gender','mobile_no')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'is_completed')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password'),
        }),
    )

    ordering = ['email']

admin.site.register(CustomUser, CustomUserAdmin)