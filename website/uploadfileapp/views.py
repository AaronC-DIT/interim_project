# Create your views here.
from django.shortcuts import render
 
# Create your views here.
from django.views import generic
from django.views.generic import CreateView
 
from uploadfileapp.models import User
 
 
class HomeView(generic.ListView):
    # name of the object to be used in the index.html
    context_object_name = 'user_list'
    template_name = 'uploadfileapp/home_page.html'
 
    def get_queryset(self):
        return User.objects.all()
 
 
# view for the user entry page
class UserEntry(CreateView):
    model = User
    fields = ['user_name', 'user_avatar']
