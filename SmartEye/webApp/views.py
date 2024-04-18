from django.shortcuts import render

def home(request):
    context = {
        'page': "home",
        'nav': True,
        'footer': True,
    }
    return render(request, "home.html", context)
