# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:27:14 2020

@author: kiddy
"""


from numpy import loadtxt
from keras.models import model_from_json

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Model Successfully Loaded from the disk file")


predictions = model.predict_classes(x)
for i in range(1,50):
    print('%s=>%d(expected %d)' %(x[i].tolist(), predictions[i],y[i]))