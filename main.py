#%% imports
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers,optimizers, losses,callbacks, applications
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

#%% 1. Data Loading
data_path = os.path.join(os.getcwd(),'dataset')

# %% 2. Data Preprocessing
BATCH_SIZE = 32
IMG_SIZE = (227,227)

dataset = keras.utils.image_dataset_from_directory(data_path,batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True)
class_names = dataset.class_names

# %% train-validation split (80% train, 20% validation)
data_batches = tf.data.experimental.cardinality(dataset)
train_dataset = dataset.skip(data_batches//5)
validation_dataset = dataset.take(data_batches//5)

# %% validation-test split (50% validation, 50% test)
val_batches = tf.data.experimental.cardinality(validation_dataset)
validation_dataset = validation_dataset.skip(val_batches//2)
test_dataset = validation_dataset.take(val_batches//2)

# %% EDA
# plot some examples
plt.figure()
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()

# %% convert datasets into prefetch datasets
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_validation = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %% 3. Create model for image augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%% Test augmentation model
for image, labels in pf_train.take(1):
    first_image = image[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')
plt.show()

# %% 4. Transfer learning
IMG_SHAPE = IMG_SIZE + (3,)

# preprocess input layer
preprocess_input = applications.mobilenet_v2.preprocess_input

# model
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# Set pretrained feature extractor as non-trainable
base_model.trainable=False
base_model.summary()
keras.utils.plot_model(base_model,show_shapes=True)

# %% Classifier
global_avg = layers.GlobalAveragePooling2D()

# output layer
output_size = len(class_names)
output_layer = layers.Dense(output_size, activation='softmax')

# Link model pipeline usine functional API
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

# Instantiate the model
model = keras.Model(inputs=inputs,outputs=outputs)
print(model.summary())

# %% Compile model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

# %% 5. Evaluate model before training
loss0, acc0 = model.evaluate(pf_validation)
print('-----Evaluation Before Training-----')
print('Loss = ', loss0)
print("Accuracy = ", acc0)

#%% 6. Callbacks and fit model
base_log_path = r'tensorboard_logs'
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_path) 
EPOCHS = 10
history = model.fit(pf_train,validation_data=pf_validation,epochs=EPOCHS,callbacks=[tb])

#%% 7. Test with test data
test_loss, test_acc = model.evaluate(pf_test)
print('-----Evaluation after training-----')
print('Test loss = ',test_loss)
print('Test Accuracy = ', test_acc)

# %% 8. Model Deployment
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)

# Stack label and prediction in one numpy array
label_vs_prediction = np.transpose(np.vstack((label_batch,y_pred)))
save_path = os.path.join("model","natasha_model.h5")
model.save(save_path)

# %% 9. Load Model
loaded_model = keras.models.load_model(save_path)
loaded_model.summary()

#%% 10. Model Analysis
true_labels = []
predicted_labels = []

for images, labels in pf_test:
  predictions = loaded_model.predict(images)
  predicted_labels_batch = np.argmax(predictions, axis=1)
  true_labels.extend(labels)
  predicted_labels.extend(predicted_labels_batch)

# Get confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
disp.plot(cmap=plt.cm.Purples)
plt.show()

# Get classification report
cr = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:")
print(cr)

# Plot some images along with their predicted labels
plt.figure(figsize=(12, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for images, labels in pf_test.take(1):
  predictions = loaded_model.predict(images)
  predicted_labels = np.argmax(predictions, axis=1)
  for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(f"Actual: {class_names[labels[i]]} \nPredicted: {class_names[predicted_labels[i]]}")
        plt.axis('off')
plt.show()