import numpy as np
data = np.load('../data.npy').item()
dataRaw = []
dataFil = []
for i in data:
    dataRaw += [1.0*data[i][1]/256,] #raw
    dataFil += [1.0*data[i][0]/256,] #filtered
dataRaw = np.stack(dataRaw)
dataFil = np.stack(dataFil)
y = np.zeros((len(data), 64))
import keras
from keras import backend as K
from inception_v3 import InceptionV3,conv2d_bn
import matplotlib.pyplot as plt
input_shape=(500,500,3)
img_input = keras.Input(shape=input_shape,name='input')
img_truth = keras.Input(shape=input_shape,name='truth')
layer1 = conv2d_bn(img_input,32,5,5,use_bias=True,name='layer1')
layer2 = keras.layers.Conv2D(32, (5, 5),
                            strides=(1,1),
                            padding='same',
                            use_bias=True,
                            name='layer2')(layer1)
layer2_plus_input = keras.layers.concatenate([img_input,layer2],name='concat')
output = keras.layers.Conv2D(3, (1, 1),
                            strides=(1,1),
                            padding='same',
                            use_bias=True,
                            name='output')(layer2_plus_input)

model = keras.models.Model(img_input, output, name='instagram_filter')
incept3 = InceptionV3(include_top=False,
                weights='imagenet',
                input_tensor=img_input,
                input_shape=(500,500,3),
                pooling=None,
                classes=1000)
for l in incept3.layers: l.trainable=False  # can't have loss function be trainable

incept_content = incept3.get_layer('activation_3').output
model_incept_content =keras.models.Model(img_input,incept_content)
input_incep_content = model_incept_content(output)
truth_incep_content = model_incept_content(img_truth)

loss = keras.layers.Lambda(lambda x: K.sqrt(K.mean((x[0]-x[1])**2, (1,2))))\
       ([input_incep_content, truth_incep_content])
final_model = keras.models.Model([img_input,img_truth],loss)
final_model.compile('adam', 'mse')
from keras.utils import plot_model
plot_model(final_model);die
final_model.fit([dataRaw,dataFil],y,8)
