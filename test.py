import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('E:/KULIAHAHAHAH/semester 5 bismillah/Studi Independet Bersertifikat (Bangkit)/project/code/Disease Tomkit/efficientnetb3-TomKit-97.94.h5')


def preprocess_image(img_path):
    img_size = (224, 224)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    # img = img / 255.0  # Normalisasi nilai pixel
    img = np.expand_dims(img, axis=0)
    return img

def get_class_name(predicted_class):
    class_names = [
        'kumato', 
        'beefsteak', 
        'tigerella', 
        'roma', 
        'japanese_black_trifele', 
        'yellow_pear',
        'sun_gold',
        'green_zebra',
        'sun_gold',
        'cherokee_purple',
        'oxheart',
        'blueberries',
        'san_marzano',
        'banana_legs',
        'german_orange_strawberry',
        'super_sweet_100'
        ]
        
    if 0 <= predicted_class < len(class_names):
        return class_names[predicted_class]
    else:
        return "Undefined"

def predict_image(img_path):
    image_preproses = preprocess_image(img_path)
    
    prediction = model.predict(image_preproses)
    predicted_class = np.argmax(prediction)
    confidence_score = np.max(prediction) * 100 
    
    class_name = get_class_name(predicted_class)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {class_name} ({confidence_score:.2f}%)")
    plt.axis('off')
    plt.show()

    print(prediction)
    print(f"Hasil Prediksi: {class_name}")
    print(f"Confidence: {confidence_score:.2f}%")
    return class_name, confidence_score


img_paths = [
    'E:/KULIAHAHAHAH/semester 5 bismillah/Studi Independet Bersertifikat (Bangkit)/project/code/Disease Tomkit/bac3.jpg'
    # 'E:/KULIAHAHAHAH/Studi Independet Bersertifikat (Bangkit)/project/New folder/tobacco-mosaic-virus-eggplant-1580133832.jpg',
    # 'E:/KULIAHAHAHAH/Studi Independet Bersertifikat (Bangkit)/project/New folder/bercak-daun-septoria.jpg',
    # 'E:/KULIAHAHAHAH/Studi Independet Bersertifikat (Bangkit)/project/New folder/bacterial-spot_tomatoes_featured.jpg',
    # 'E:/KULIAHAHAHAH/Studi Independet Bersertifikat (Bangkit)/project/New folder/images.jpg'
]

for img_path in img_paths:
    predict_image(img_path)
