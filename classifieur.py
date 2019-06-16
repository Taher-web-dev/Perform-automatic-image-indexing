#!usr/bin/python3.6
#encoding:utf_8
import os
from PIL import Image
from PIL.ImageOps import equalize
from PIL import ImageFilter
import numpy as np 
import pandas as pd 
import cv2 as cv
import pickle
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
path=os.path.abspath(os.path.dirname(__file__))
image=input("entrer le nom de fichier de l'image à classifier : ")
image=os.path.join(path,image)
# method to use for classification.
print("Entrer la méthode de classification à utiliser:")
choix=input("Entrer 1 pour utiliser l'approche classique.\nEntrer 2 pour utiliser la méthode CNN.")
choix=int(choix)
if choix==1:
    image=Image.open(image)
#Eliminate noising
    image=image.filter(ImageFilter.GaussianBlur(radius=6))
#Enhance image contrast by equialization.
    image=equalize(image)
#extract features and keypoints detector of the image
    grey_image=cv.cvtColor(np.array(image),cv.COLOR_BGR2GRAY)
    sift=cv.xfeatures2d.SIFT_create()
    kp,desc=sift.detectAndCompute(grey_image,None)
#bag of features of the image
    with open('donnes','rb') as fichier :
       mon_depickler=pickle.Unpickler(fichier)
       lr=mon_depickler.load()
       model=mon_depickler.load()
    image_features=np.zeros((1,500))
    words=model.predict(desc)
    for w in words:
       image_features[0,w]+=1
    race=lr.predict(image_features)
    print('La race du chien de la photo d\'entrée est : {}'.format(race))
if choix==2:
   chemin=os.path.join(path,'cnn')
   model=load_model(chemin)
   img=load_img(image,target_size=(224,224,3))
   img=img_to_array(img)
   img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
   pr=model.predict(img)
   ind=np.where(pr>0)[0]
   ch=os.path.join(path,'cl')
   with open(ch,'rb') as fi :
        depickler=pickle.Unpickler(fi)
        classes=depickler.load()
   for lb,i in classes.items():
       if i==ind :
           lab=lb.split('-')[1]
           break
   print('La race du chien de la photo d\'entrée est : {}'.format(lab))
else :
   print("Au revoir !")
