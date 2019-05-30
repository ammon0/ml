

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
from matplotlib import pyplot 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.layers import LeakyReLU, ThresholdedReLU
from keras import optimizers
BS=128
EPOCHS=60000
PATIENCE=200
train = pd.read_csv('/home/itadmin/ml/images/cropped_data.csv', index_col=0)
trainDir = '/home/itadmin/ml/images/training/resized-images/'
print(train.head())
#print(train.columns)
count = 0
train_image = []
for i in tqdm(range(train.shape[0])):
    print(train['userid'][i])
    try:
        img = image.load_img(trainDir + train['userid'][i] + '.jpg', target_size=(64,64,3))
        count+=1
    except:
        print('image not found!')
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
print('%d images loaded'%count)

X = np.array(train_image)
#print(X.shape)
#print(train['gender'][2])
y =np.array(train.drop(['userid','age', 'ope', 'con', 'ext', 'agr', 'neu'], axis=1))
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.08)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', input_shape=(64,64,3)))
#model.add(Activation('sigmoid'))
#model.add(ThresholdedReLU(1e-6))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2) ))
model.add(BatchNormalization())
#model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
#model.add(Activation('sigmoid'))
#model.add(ThresholdedReLU(1e-6))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
#model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
#model.add(Activation('sigmoid'))
#model.add(ThresholdedReLU(1e-6))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
#model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
#model.add(Activation('sigmoid'))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=256, kernel_size=(2,2), padding='same'))
#model.add(Activation('sigmoid'))
model.add(LeakyReLU(alpha=0.3))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
'''
model.add(Conv2D(filters=256, kernel_size=(2,2)))
#model.add(Activation('sigmoid'))
model.add(LeakyReLU(alpha=0.3))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
'''
model.add(Flatten())

model.add(Dense(256))
#model.add(Activation('sigmoid'))
#model.add(ThresholdedReLU(1e-6))
#model.add(LeakyReLU(alpha=0.05))
#model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

print(model.summary())
opt = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

filepath='/home/itadmin/ml/images/weights/gender-weights-{epoch:02d}-{val_acc:.2f}.h5'

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
checkpointer=ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



aug=ImageDataGenerator(rotation_range=90, zoom_range=[0.5, 1.0], width_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
reduceLR=ReduceLROnPlateau(monitor='val_loss', factor=0.67, patience=PATIENCE//3, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-6)
callbacks=[checkpointer, es, reduceLR]

history=model.fit_generator(aug.flow(X_train, y_train, batch_size=BS), validation_data=(X_test, y_test), steps_per_epoch=len(X_train)//BS, epochs=EPOCHS, callbacks=callbacks)

#model.train_on_batch(X,y)
#model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), batch_size=32, callbacks=[checkpointer])

#model.save_weights('gender_and_age_50.weights')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

scores=model.evaluate(X, y, verbose=1)
print(scores)
