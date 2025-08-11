import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
your_dataset_directory = ''
BATCH_SIZE = 32
#80 Epochs was enough for me to get to mid 90's in accuracy
EPOCHS = 80
input_shape = (256,256,3)

dataset = tf.keras.utils.image_dataset_from_directory(your_dataset_directory, image_size=(256,256), batch_size=BATCH_SIZE, label_mode='int', shuffle=True)

class_names = dataset.class_names
classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), (2,2), activation='relu',input_shape=input_shape,padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(64, (5,5), (2,2), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(128, (5,5), (2,2), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(256, (5,5), (2,2), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(classes,activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#optional, if you want to load the model after it's been trained
#model = tf.keras.models.load_model('models')

model.fit(dataset, epochs=EPOCHS)
def get_random_image():
    for images, labels in dataset.take(1):

        idx = np.random.randint(0, images.shape[0])

        random_image = images[idx].numpy()
        random_label = labels[idx].numpy()

        break

    random_class_name = class_names[random_label]
    return random_image, random_class_name

def prediction():
    image, class_name = get_random_image()
    img_batch = tf.expand_dims(image, 0)
    pred_probs = model.predict(img_batch)
    predicted_class_index = np.argmax(pred_probs[0])
    predicted_class_name = class_names[predicted_class_index]
    img_to_show = image / 255.0
    print('real: ', class_name)
    
    print('predicted: ', predicted_class_name)
    plt.imshow(img_to_show)
    plt.axis('off')
    plt.show()

    return
model.save('models')

prediction()
