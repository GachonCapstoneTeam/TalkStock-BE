from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response


# Create your views here.

@api_view()
def hello_world(request):
    return Response({"originaltext" : "하늘이 장차 그 사람에게 큰 사명을 주려 할 때는 반드시 먼저 그의 마음과 뜻을 흔들어 고통스럽게 하고,  힘줄과 뼈를 굶주리게 하여 궁핍하게 만들어 그가 하고자 하는 일을 흔들고 어지럽게 하나니그것은 타고난 작고 못난 성품을 인내로써 담금질을 하여 하늘의 사명을 능히 감당할 만 하도록 그 기국과 역량을 키워주기 위함이다."})