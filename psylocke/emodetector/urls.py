from django.conf.urls import patterns, url
from emodetector import views

urlpatterns = patterns('',
        url(r'^$', views.index, name='index'))