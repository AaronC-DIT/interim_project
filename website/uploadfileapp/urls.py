from django.conf.urls import url
 
from uploadfileapp import views
 
app_name = 'uploadfileapp'
 
urlpatterns = [
 
    # uploadfileapp/
    url(r'^$', views.HomeView.as_view(),name='home' ),
 
    # uploadifileapp/register
    url(r'^register/$', views.UserEntry.as_view(), name='user-entry'),
 
]
