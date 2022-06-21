import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import warnings
from keras_preprocessing import image
warnings.filterwarnings('ignore')

print(0)
val_path="Vegetable Images/validation"
train_path="Vegetable Images/train"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_path,
                                                               seed=2509,
                                                               image_size=(224, 224),
                                                              batch_size=32)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(val_path,
                                                              seed=2509,
                                                              image_size=(224, 224),
                                                              shuffle=False,
                                                              batch_size=32)


class_names = train_dataset.class_names

model = keras.models.load_model("vegetable")

print("========================================\n")
model.evaluate(val_dataset)
print("========================================\n")
again = "Y"
while again == "Y" or  again == "y":
    vegie = input("Enter Vegetable Name: ")
    no = input("Enter image number: ")
    image_path="Vegetable Images/test/"+vegie+"/"+no+".jpg"
    try:
        img = image.load_img(image_path, target_size=(224,224,3))
    except:
        print("Image might not exist, try checking name or number")
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = model.predict(images, batch_size=32)
    label = np.argmax(pred, axis=1)
    print("Actual: "+image_path.split("/")[-2])
    print("Predicted: "+class_names[np.argmax(pred)])
    plt.imshow(img)
    plt.show()
    again = input("Again?? ")
    print("====================================")





