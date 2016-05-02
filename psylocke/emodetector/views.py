from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from PIL import Image
import json
# Create your views here.

from django.views.decorators.csrf import csrf_exempt
from FaceRecognizer import FaceRecognizer

@csrf_exempt
def index(request):
    if request.method == 'GET':
        print request.GET
        fc = FaceRecognizer()
        res = fc.getClassification()
        return JsonResponse({'foo':res})
        #return HttpResponse("Hello face")
    else:
        print ("get post")
        print (request.FILES)
        f = request.FILES['file']
        im = Image.open(f)
        im.save('photo.png')
        return JsonResponse({'foo':'bar'})