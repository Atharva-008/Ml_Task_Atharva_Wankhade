import tensorflow as tf 
from tensorflow.keras import datasets,layers,models
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
train_images,test_images=train_images/255.0,test_images/255.0
#print(train_images)
model=models.Sequential([
layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
layers.MaxPooling2D((2,2)),
layers.Conv2D(64,(3,3),activation='relu'),
layers.MaxPooling2D((2,2)),
layers.Conv2D(64,(3,3),activation='relu'),
layers.Flatten(),
layers.Dense(64,activation='relu'),
layers.Dense(10,activation='softmax')])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from tensorflow.keras.utils import to_categorical
train_labels_one_hot=to_categorical(train_labels)
test_labels_one_hot=to_categorical(test_labels)
model.fit(train_images,train_labels_one_hot,epochs=10,validation_data=(test_images,test_labels_one_hot))
test_loss,test_acc=model.evaluate(test_images,test_labels_one_hot,verbose=2)
print(f'Test Accuracy -> {test_acc}')
