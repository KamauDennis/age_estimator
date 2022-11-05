import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import os
from zipfile import ZipFile

with ZipFile('UTKFACE.zip','r') as f:
    f.extractall()

filepaths = []
dir = 'UTKFace'
for root,subdirs,files in os.walk(dir):
    if dir in subdirs:
      subdirs.remove(dir)
    for file in files:
        filepath = os.path.join(root,file)
        filepaths.append(filepath)   

image_count = len(filepaths)           

#create a file dataset
filepath_dataset = tf.data.Dataset.list_files(filepaths,shuffle=True)
filepath_dataset = filepath_dataset.shuffle(image_count,reshuffle_each_iteration=False)

#splitting the dataset 
val_size = int(image_count*0.1)
training_ds = filepath_dataset.skip(val_size)
val_ds = filepath_dataset.take(val_size)
test_size = int(image_count*0.05)
train_ds = training_ds.skip(test_size)
test_ds = training_ds.take(test_size)
print("Train_ds size ",train_ds.cardinality().numpy())
print("Test_ds size ",test_ds.cardinality().numpy())
print("Val_ds size ",val_ds.cardinality().numpy())

def find_labels(y):
    lst =tf.constant(['0-1','2-5','6-12','13-19','20-29','30-39','40-59','60-120']) 
    label_lst =tf.TensorArray(tf.string, size=1, dynamic_size=False, clear_after_read=True)
    for x in lst:
        lower_lim = tf.strings.to_number(tf.strings.split(x,sep='-')[0],tf.int32)
        upper_lim = tf.strings.to_number(tf.strings.split(x,sep='-')[1],tf.int32)
        if (y>=lower_lim) and (y<=upper_lim):
            label_lst = label_lst.write(0,x)
    label = label_lst.read(0)
    #print(label)
    return label 

class_names =tf.constant(['0-1','2-5','6-12','13-19','20-29','30-39','40-59','60-120']) 

#a function that encodes the labels to onehot
def encode_labels(label):
    indices = tf.range(len(class_names),dtype=tf.int32)
    table_init = tf.lookup.KeyValueTensorInitializer(class_names,indices)
    table = tf.lookup.StaticHashTable(table_init,default_value=-1)
    label_indices = table.lookup(label)
    label_one_hot = tf.one_hot(label_indices,depth=len(class_names),dtype=tf.int64)
    return label_one_hot

def get_label(file_path):
    class_ = tf.strings.split(file_path,os.path.sep)[1]
    age_ = tf.strings.split(class_,'_')[0]
    gender_ = tf.strings.split(class_,'_')[1]
    race_ = tf.strings.split(class_,'_')[2]
    gender = tf.strings.to_number(gender_,tf.int32)
    age = tf.strings.to_number(age_,tf.int32)
    label_age = find_labels(age)
    one_hot = encode_labels(label_age)
    return tf.argmax(one_hot)    

#define some parameters 
batch_size = 32
img_height = 200
img_width = 200

def decode_img(img):
    #convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img,channels=3)
    return tf.image.resize(img,[img_height,img_width])

def process_path(file_path):
    label = get_label(file_path)
    #load the raw data from the path as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img,label

#use Dataset.map to create a dataset of image,label pair
#set number of parallel calls so multiple images are loaded/processed in parallel
train_ds = train_ds.map(process_path,num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(process_path,num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(process_path,num_parallel_calls=tf.data.AUTOTUNE)

def configure_for_performance(ds):
    #ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return(ds)

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

#Let's first create a ResidualUnit layer
class ResidualUnit(keras.layers.Layer):
    def __init__(self,filters,strides=1,activation='relu',**kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters,3,strides=strides,padding='same',use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters,3,strides=1,padding='same',use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters,1,strides=strides,padding='same',use_bias=False),
                keras.layers.BatchNormalization()
            ]
        
    def call(self,inputs):
        Z =inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)  

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64,7,strides=2,input_shape=[200,200,3],padding='same',use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same'))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters,strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(8,activation='softmax'))

#compiling the model
model.compile(loss=["sparse_categorical_crossentropy"],
             optimizer=keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999),
             metrics='accuracy')

#training the model
history = model.fit(train_ds,epochs=40,validation_data=val_ds)    

#saving the model in tf format
model_version = "0001"
model_name = "age_detector_model"
model_path = os.path.join(model_name,model_version)
tf.saved_model.save(model,model_path)