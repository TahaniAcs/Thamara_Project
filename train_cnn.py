import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# 1. تحديد المسارات (المعدلة للعمل مع Kaggle Cache)
# --------------------------------------------------------------------------
# المسار الأساسي (Cache Path) الذي تم طباعته من kagglehub
import os
import tensorflow as tf # تأكد من أن هذا موجود في البداية

# المسار الأساسي (Cache Path) الذي تم طباعته من kagglehub
BASE_PATH = r"C:\Users\Lenovo\.cache\kagglehub\datasets\leftin\fruit-ripeness-unripe-ripe-and-rotten\versions\2"

# المسار المعدل ليناسب التسلسل الهرمي الكامل الذي وجدته:
# fruit_ripeness_dataset / archive (1) / dataset / train
TRAINING_DIR = os.path.join(BASE_PATH, "fruit_ripeness_dataset", "archive (1)", "dataset", "train")
VALIDATION_DIR = os.path.join(BASE_PATH, "fruit_ripeness_dataset", "archive (1)", "dataset", "test")
# --------------------------------------------------------------------------


# معلمات النموذج (Hyperparameters)
IMAGE_SIZE = (128, 128) 
BATCH_SIZE = 32
NUM_CLASSES = 9 
EPOCHS = 50 


# 1. تجهيز البيانات 
# --------------------------------------------------------------------------
# مُولِّد بيانات التدريب (مع تحسين البيانات)
train_datagen = ImageDataGenerator(
    rescale=1./255, # تطبيع البكسلات
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

# مُولِّد بيانات التحقق (التطبيع فقط)
validation_datagen = ImageDataGenerator(rescale=1./255) 

# تجهيز مُولِّد بيانات التدريب
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# تجهيز مُولِّد بيانات التحقق
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
# --------------------------------------------------------------------------


# 2. بناء نموذج CNN (شبكة عصبونية تلافيفية)
# --------------------------------------------------------------------------
model = Sequential([
    # الطبقة التلافيفية الأولى
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # الطبقة التلافيفية الثانية
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # الطبقة التلافيفية الثالثة
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # التسطيح والإخراج
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
# --------------------------------------------------------------------------


# 3. تجميع وتدريب النموذج
# --------------------------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("بدء تدريب النموذج...")
# يجب أن يظهر هنا رسالة "Found XXXX images belonging to 9 classes."
history = model.fit(
    train_generator,
    epochs=EPOCHS, 
    validation_data=validation_generator,
    verbose=1
)
# --------------------------------------------------------------------------


# 4. حفظ النموذج لاستخدامه في تطبيق الجوال
# --------------------------------------------------------------------------
model.save('thamara_ripeness_cnn_model.h5')
print("تم حفظ النموذج بنجاح.")