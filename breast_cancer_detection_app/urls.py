from .views import breast_cancer_detection
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('',breast_cancer_detection,name='index'),
]
