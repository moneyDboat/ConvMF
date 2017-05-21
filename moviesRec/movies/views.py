# -*- coding: utf-8 -*-
# 上面这行注释是要输进去的，这样的话解析才可以出现中文，不然会报错，原因在后面
from django.shortcuts import render
from django.http import HttpResponse
import json
import os
import random
import numpy as np


# Create your views here.
def index(request):
	if request.method == 'POST':
		print(request.body)
		json_data = json.loads(request.body)
		user = json_data['user']
		result = predict(user)
		return HttpResponse(json.dumps(result))
	else:
		return HttpResponse('haha')


def predict(user):
	# load data
	module_path = os.path.dirname(__file__)
	U = np.loadtxt(module_path + '/U.dat')
	V = np.loadtxt(module_path + '/V.dat')

	approx_R = U[user].dot(V.T)
	n_range = range(len(approx_R))
	R_dict = dict(zip(n_range, approx_R))
	R_sort = sorted(R_dict.items(), key=lambda d:d[1], reverse=True)

	R_comm = [tupe[0] for tupe in random.sample(R_sort[:30], 10)]
	result = {'recommend':R_comm}
	return result







