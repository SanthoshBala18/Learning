from sklearn.externals import joblib
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Dense,Softmax

from keras.utils import to_categorical


(features,labels),(test_features,test_labels) = mnist.load_data()

num_pixel = features.shape[1]*features.shape[2]
features = features.reshape(features.shape[0],num_pixel).astype('float32')

test_features = test_features.reshape(test_features.shape[0],num_pixel).astype('float32')

labels = to_categorical(labels)
test_labels = to_categorical(test_labels)
num_classes = test_labels.shape[1]

features = features/255
test_features = test_features/255

model = Sequential()

model.add(Dense(num_pixel,input_dim=num_pixel))
model.add(Dense(25,activation="relu"))
model.add(Dense(num_classes,activation="softmax"))

model.compile(optimizer="Adam",loss="categorical_crossentropy")
model.fit(features,labels,epochs=3)

'''model = SVC()
model.fit(features,labels)'''

print(model.evaluate(test_features,test_labels,batch_size=128))

joblib.dump(model,"keras_mdoel",compress=3)