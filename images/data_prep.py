'''
    This program detects faces in images in the folder '/tcss455/training/image/'
    and crops everything else out and resizes the image to 128 x 128.
    
    Matthew Skipworth
'''
import pandas
import os
import numpy as np
import cv2
import csv

HEIGHT = 64
WIDTH = 64

dimensions = (WIDTH, HEIGHT)

TRAINING_DATA = 1


### file directories for training and test images on VM; original, cropped and resized
trainImgReadDirectory = '/home/itadmin/ml/images/training/image/'
trainCroppedDirectory = '/home/itadmin/ml/images/training/cropped-images/'
trainImgWriteDirectory = '/home/itadmin/ml/images/training/resized-images/'
'''
testImgReadDirectory = '/Users/Matthew/SCHOOL/UWtacoma/TCSS455/tcss455/public-test-data/image/'
testCroppedDirectory = '/Users/Matthew/SCHOOL/UWtacoma/TCSS455/tcss455/public-test-data/cropped-images/'
testImgWriteDirectory = '/Users/Matthew/SCHOOL/UWtacoma/TCSS455/tcss455/public-test-data/resized-images/'
'''

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if (face_cascade is None):
    print('face_cascade not working!')
    quit()
'''
eye_cascade = cv2.CascadeClassifier('/opencv/data/haarcascades/haarcascade_eye.xml')
if (eye_cascade is  None):
    print('eye_cascade not working!')
    quit()

'''




### function to detect faces and crop faces

def detectAndCrop():
    noface=list()
    if TRAINING_DATA:
        for filename in os.listdir(trainImgReadDirectory):
            if filename.endswith('.jpg'):
                try:
                    img=cv2.imread((trainImgReadDirectory + str(filename)))
                except:
                    print('invalid operation!')
            
            #print('before graying')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                if (gray is None):
                    print('gray not working!')
                    quit()
                    
                try:
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5, flags=cv2.CASCADE_SCALE_IMAGE)
                    if faces == ():
                    	noface.append(filename)
                    	print('face not found')
 
                except:
                    print('faces not working!')
                    

            
                for (x,y,w,h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]
                    cv2.imwrite(trainCroppedDirectory + filename, gray[y:((y-50)+(h+50)), x:((x-50)+(w+50))])

    else:
        
        for filename in os.listdir(testImgReadDirectory):
            if filename.endswith('.jpg'):
                try:
                    img=cv2.imread((testImgReadDirectory + str(filename)))
                except:
                    print('invalid operation!')

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if (gray is None):
                    print('gray not working!')
                    quit()
                
                try:
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5, flags=cv2.CASCADE_SCALE_IMAGE)
                    for (x,y,w,h) in faces:
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = img[y:y+h, x:x+w]
                        cv2.imwrite(testCroppedDirectory + filename, gray[y:((y-50)+(h+50)), x:((x-50)+(w+50))])
                except:
                    print('faces not working!')


                
    return noface

### function to resize images to 64 x 64

def resize():

    if TRAINING_DATA:
        for filename in os.listdir(trainCroppedDirectory):
            if filename.endswith('.jpg'):
                try:
                    img=cv2.imread((trainCroppedDirectory + str(filename)))
                    resized=cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(trainImgWriteDirectory+filename, resized)
                except:
                    print('invalid operation! ...bad training data')

    else:
        for filename in os.listdir(testCroppedDirectory):
            if filename.endswith('.jpg'):
                try:
                    img=cv2.imread((testCroppedDirectory + str(filename)))
                    #print(img.shape)
                    resized=cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(testImgWriteDirectory+filename, resized)
                except:
                    print('invalid operation!')

def modify_csv(fileList):
    
    df = pandas.read_csv('/home/itadmin/ml/images/training/profile/profile.csv', index_col=0)
    df.set_index('userid', inplace=True)
    for filename in fileList:
        user=filename.split('.')[0]
        print(user)
        df=df.drop(user)
    print(df)    
    df.reset_index(inplace=True)
    df.to_csv('cropped_data.csv')
        
'''
#fieldnames=['userid', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu']
newTableWriter = csv.writer(newTable, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
newdf=pandas.DataFrame()


df.set_index('userid', inplace=True)
for filename in os.listdir(trainCroppedDirectory):
user=filename.split('.')[0]
#newTableWriter.writerow(df.loc[user,:])
newdf.append(df.loc[user,:])

print(df.loc[user,:])
print(newdf.head())    
newdf.to_csv('cropped_data.csv')    
'''

### main function

def main():
    print('Detecting faces...')
    noface=detectAndCrop()
    print(len(noface))
    print('Done detecting!')
    print('Resizing images..')
    resize()
    print('Done resizing!')
    modify_csv(noface)
    #cv2.destroyAllWindows()
main()




