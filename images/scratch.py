import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from tqdm import tqdm
from joblib import dump, load

train = pd.read_csv('/data/training/profile/profile.csv', index_col=0)
trainDir = '/data/training/image/'


print(train.head())
print(train.columns)
count = 0
train_image = []
for i in tqdm(range(train.shape[0])):
#print(train['userid'][i])
    try:
        img = image.load_img((trainDir + train['userid'][i] + '.jpg'), target_size=(128,128,3))
        count+=1
    except:
        print('image not found!')
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
print('%d images loaded'%count)

gender=train['gender']
X = np.array(train_image)
print(X.shape)
print(train['gender'][2])
y=np.array(train.drop(['userid','age', 'ope', 'con', 'ext', 'agr', 'neu'], axis=1))
#y=np.array(['0.0', '0.1'])
print('y_shape:')
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)



model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3),strides=(2,2), activation="relu", input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(3, 3),strides=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3, 3),strides=(2,2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
'''
model.add(Conv2D(filters=64, kernel_size=(3, 3),strides=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
'''
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])

filepath="/home/itadmin/ml/images/weights/gender-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, callbacks=[checkpoint])


dump(model, 'gender.weights')
scores=model.evaluate(X, y, verbose=0)
