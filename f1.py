import plaidml.keras
plaidml.keras.install_backend()
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import numpy as np 
import os 
import cv2 
import pandas 
import matplotlib.pyplot as plt
import glob
from keras.utils import np_utils
from random import shuffle 

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from keras.layers import Conv2D
from keras.layers import MaxPool2D, Dense, Flatten  
from keras.layers import Dense, Dropout, Activation
import os
import numpy as np
import cv2

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames
    
 




vailed_ext = [".jpg",".png",".pgm"]
import os 

images = []
labels =[]

c=0
def load_dataset(rootDir,c=0):   
    
    
    for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        filename, file_extension = os.path.splitext(path) 
        if file_extension in vailed_ext:
            
            image = load_img(path, target_size=(100, 100))
            '''image= detect_faces(image)'''
            images.append(img_to_array(image))
            labels.append(c)
            print("classes",c)
                 
            
            #x= cv2.resize(x, (100,100), interpolation = cv2.INTER_AREA);
            print(path)             
            
            
        if os.path.isdir(path):
           c=c+1        
           load_dataset1(path,c-1)
           
           
        
        
    print("Total classes",c)
        
    
    return images,labels, c


    
            
            
		 
  
         
         
def make_network(X_train,classes):
    cnn = Sequential()

    cnn.add(Conv2D(activation='tanh',strides=(1,1),kernel_size=(9,9),filters=16,padding='same',input_shape=X_train.shape[1:]))
    cnn.add(MaxPool2D(strides=(1,1),pool_size=(2,2),padding='same'))

    cnn.add(Conv2D(activation='tanh',strides=(1,1),kernel_size=(7,7),filters=16,padding='same'))
    cnn.add(MaxPool2D(strides=(2,2),pool_size=(2,2),padding='same'))

    cnn.add(Conv2D(activation='tanh',strides=(1,1),kernel_size=(5,5),filters=32,padding='same'))
    cnn.add(MaxPool2D(strides=(2,2),pool_size=(2,2),padding='same'))

    cnn.add(Conv2D(activation='tanh',strides=(1,1),kernel_size=(3,3),filters=32,padding='same'))
    cnn.add(MaxPool2D(strides=(2,2),pool_size=(2,2),padding='same'))

    cnn.add(Flatten())

    cnn.add(Dense(activation='relu',units=2048))

    cnn.add(Dense(activation='softmax',units=classes))

    return cnn


if __name__ == '__main__':


    ###########AT&T Dataset#################
    
    images,labels, nb_classes = load_dataset('At&t',0)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    


    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)


    X_train=np.array(X_train)
    X_valid=np.array(X_valid)
    X_test=np.array(X_test)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_valid /= 255
    X_test /= 255
    
    model = make_network(X_train,nb_classes)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    hist1 = model.fit(X_train, Y_train, batch_size=32, epochs=150, validation_data=(X_valid,Y_valid),
                           shuffle=True)
    score=model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100)) 
    
    ###################################
    
   
   