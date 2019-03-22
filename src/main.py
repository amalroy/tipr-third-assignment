import re
import argparse
import mnist_reader
import numpy as np
import keras
import pickle
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation, BatchNormalization
from keras.utils.generic_utils import get_custom_objects
from keras.models import Sequential
from keras.models import load_model
from keras import backend as K

from sklearn.metrics import accuracy_score,f1_score
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def swish(x):
    return (K.sigmoid(x) * x)
get_custom_objects().update({'swish': swish})
def keras_nn(input_shape):
    net=Sequential()
    net.add(Flatten())
    net.add(Dense(512, activation='relu', input_shape=(input_shape,)))
    net.add(Dropout(0.2))
    net.add(Dense(512, activation='relu'))
    net.add(Dropout(0.2))
    net.add(Dense(num_classes, activation=tf.nn.softmax))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    net.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return net
def keras_cnn(input_shape,n_fil,act,init='lecun_normal'):
    net=Sequential()
    net.add(Conv2D(n_fil[0], (4, 4), padding='same',input_shape=input_shape,kernel_initializer=init))
    net.add(BatchNormalization())
    #net.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    for i in range(len(n_fil)):
        if(i == len(n_fil)-1):
            net.add(Dropout(0.25))
            net.add(Flatten())
            net.add(Dense(256,activation='elu',kernel_initializer=init))
            net.add(Dropout(0.5))
            net.add(Dense(32,activation='tanh',kernel_initializer=init))
            net.add(Dropout(0.5))
            net.add(Dense(10,activation=tf.nn.softmax,kernel_initializer=init))
        else:
            net.add(Conv2D(n_fil[i+1],(3,3), padding='same',activation=act,kernel_initializer=init))
            net.add(BatchNormalization())
            if(i % 1 == 0):
                net.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    
    #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    opt = keras.optimizers.Adadelta()
    #opt = keras.optimizers.Adam()
    net.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    return net
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data')
    parser.add_argument('--train-data')
    parser.add_argument('--filter-config',type=str)
    parser.add_argument('--activation')
    parser.add_argument('--dataset')
    args=parser.parse_args()
    mode='train'
    if(args.train_data == None):
        mode='test'
    if(args.dataset == 'CIFAR-10'):
        n_classes=10
        folder=args.test_data
        #assuming the test file is in same format saved as data_test
        fname='test_batch'
        test=unpickle(folder+fname)
        tmp_images=test[b'data'].reshape([-1, 3, 32, 32])
        X_test=tmp_images.transpose([0, 2, 3, 1])
        y_test=np.array(test[b'labels'])
        #y_test=keras.utils.to_categorical(y_test, n_classes)
        X_test=X_test/255
    else:
        n_classes=10
        X_test,y_test=mnist_reader.load_mnist(args.test_data, kind='t10k')
        #y_test=keras.utils.to_categorical(y_test, n_classes)
        X_test=X_test.reshape([-1,28,28,1])
        X_test=X_test/255
        
    modelfname='model_'+args.dataset+'.h5' 
    if (mode =='test'):
        #network=pickle.load(open(modelfname, 'rb'))
        network=load_model(modelfname)
        y_pred=np.argmax(network.predict(X_test),axis=1)
        acc=accuracy_score(y_test,y_pred)
        f1_mic=f1_score(y_test,y_pred,average='micro')
        f1_mac=f1_score(y_test,y_pred,average='macro')
        print("Test Accuracy ::",acc)
        print("Test Macro F1-score ::",f1_mac)
        print("Test Micro F1-score ::",f1_mic)
    if(mode == 'train'):
        if(args.dataset == 'CIFAR-10'):
            n_classes=10
            folder=args.train_data
            for i in range(1,6):
                fname='data_batch_'+str(i)
                batch=unpickle(folder+fname)
                tmp_images=batch[b'data'].reshape([-1, 3, 32, 32])
                if i==1 :
                    X_train=tmp_images.transpose([0, 2, 3, 1])
                    y_train=np.array(batch[b'labels'])
                else:
                    X_train=np.vstack((X_train,tmp_images.transpose([0, 2, 3, 1])))
                    y_train=np.append(y_train,np.array(batch[b'labels']))
            # to one hot
            y_train=keras.utils.to_categorical(y_train, n_classes)
            X_train=X_train/255
        else:
            n_classes=10
            X_train,y_train=mnist_reader.load_mnist(args.test_data, kind='train')
            y_train=keras.utils.to_categorical(y_train, n_classes)
            X_train=X_train.reshape([-1,28,28,1])
            X_train=X_train/255
        if(args.filter_config != None):
            config=re.sub(' ',',',args.filter_config)
            config=re.sub('\[|\]','',config)
            n_fil=np.fromstring(config,dtype=int,sep=',')
        else:
            n_fil=[8,16,16]
        if(args.activation != None):
            act=args.activation
        else:
            act='relu'
        input_shape=X_train.shape[1:]
        conv=keras_cnn(input_shape,n_fil,act)
        print(conv.summary())
        max_train=int(X_train.shape[0]*0.9)
        X_val=X_train[max_train:]
        y_val=y_train[max_train:]
        batch_size=64
        epochs=30
        conv.fit(X_train[:max_train], y_train[:max_train], batch_size=batch_size,epochs=epochs,shuffle=True,
                validation_data=(X_val,y_val))                  
        save=True
        if(save==True):
            #pickle.dump(conv, open(modelfname, 'wb'))
            conv.save(modelfname)
        print("training finished")
        y_pred=np.argmax(conv.predict(X_test),axis=1)
        acc=accuracy_score(y_test,y_pred)
        f1_mic=f1_score(y_test,y_pred,average='micro')
        f1_mac=f1_score(y_test,y_pred,average='macro')
        print("Test Accuracy ::",acc)
        print("Test Macro F1-score ::",f1_mac)
#    if (args.train_data != None):
#       mode='train'
        
   
