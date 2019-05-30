import keras
import h5py
import os
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm

IMG_DIMENSIONS=(64, 64)

weightsDir='/home/itadmin/ml/images/training/weights/'

genderClassifier=load_model(weightsDir+'gender-weights-1292-0.88.h5')

imageDir='/home/itadmin/ml/images/training/image/'

profiles = pd.read_csv('/home/itadmin/ml/images/training/profile/profile.csv', index_col=0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

multiFaces=0
noFaces=0
unused=0

def detectFace(theImage):
    global multiFaces
    global noFaces
    result=0
    try:
        img=cv2.imread(theImage)
    except:
        print('invalid operation!')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (gray is None):
        print('gray not working!')
        quit()
    try:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, flags=cv2.CASCADE_SCALE_IMAGE)
#print('face not found')
    except:
        print('faces not working!')

    if (faces != ()):
        #print(faces)
        result=1
    if(len(faces)>1):
        multiFaces+=1
    elif(len(faces)<1):
        noFaces+=1
    return len(faces)

def cropAndResize(theImage):
    try:
        #print(theImage)
        img = cv2.imread(theImage)
    except:
        print('invalid operation!')
        quit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    except:
        print('faces not working!')
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        resized=cv2.resize(roi_gray, IMG_DIMENSIONS, interpolation = cv2.INTER_AREA)
        return resized

def main():
    global noFaces
    global multiFaces
    global unused
    profiles.set_index('userid', inplace=True)
    profiles.sort_index(inplace=True) ### sort the profiles by their id number
    profiles.reset_index(inplace=True)
    result=pd.DataFrame(columns=['userid', 'gender'])
    result['userid']=profiles['userid']
    result.set_index('userid', inplace=True)
    result.sort_index()
    count = 0
    for i in tqdm(range(profiles.shape[0])):
        thisUser=profiles['userid'][i]
        
        try:
            #img = image.load_img(imageDir + thisUser + '.jpg', target_size=(64,64,3))
            count+=1
        except:
            print('image not found!')

        if ((detectFace(imageDir + thisUser + '.jpg') < 1) or (detectFace(imageDir + thisUser + '.jpg') > 1)):
            ### use BASELINE
            #print('using baseline')
            unused+=1
            result.at[thisUser, 'gender'] = np.ones(1)

    #result.at[thisUser, 'gender'] = (result.at[thisUser, 'gender']).astype(float)

    #print(result.at[thisUser, 'gender'])
        else:
            img = cropAndResize(imageDir + thisUser + '.jpg')
            img = image.load_img(imageDir + thisUser + '.jpg', target_size=(64,64,3))
            img = image.img_to_array(img)
            img = img/255
            img = img.reshape(1,64,64,3)
            result.at[thisUser, 'gender']=(genderClassifier.predict_classes(img))[0].astype(float)
    #print(result.at[thisUser, 'gender'])
    print(result)
    #result['gender']=result['gender']### convert these elements to float
    
    print('%d images loaded'%count)
    print('%d images without a face'%noFaces)
    print('%d images with multiple faces'%multiFaces)
    print('%d images fed to model'%(count-unused))
    print(metrics.accuracy_score(profiles['gender'], result['gender'].astype(int)))

main()
