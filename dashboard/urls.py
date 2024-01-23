from django.urls import path, include
from .views import *
from analysis.compare_bill import CompareBillionaires

urlpatterns = [
    path("", main, name="main-home"),

    path('upload/', upload_file, name='main-upload_file'),
    path('user/files/', all_user_files, name='main-user_files'),

    path('d/', BillionairesView, name='main-bil'),
    path('d/compare', CompareBillionaires.as_view(), name='main-bil-cmp'),
    path('api/d/', BillionairesSearch.as_view(), name='main-bil-api'),
    path("d/wishlist", BilWishList, name="main-bil-wishlist"),

    path("all_users/", AllUsersView, name="main-users-all"),
    path("user-create", UserAddView.as_view(), name="main-user-add"),

    path("business-growth", BusinessGrowthcls, name="main-business-growth"),
    path("quiz-page", QuizPage, name="main-quiz-page"),

    # Data Export
    path("export-excel-all-users/", UserDataExportExcel, name="ctm_admin-export-excel-all-users"),
    path("export-csv-all-users/", UserDataExportCsv, name="ctm_admin-export-csv-all-users"),
]