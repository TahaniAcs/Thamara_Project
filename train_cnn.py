import os
import json
import tensorflow as tf
import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ==========================
# 1) إعداد المسارات (ديناميكي)
# ==========================

print("⏳ جاري التحقق من البيانات عبر KaggleHub...")
# تحميل البيانات تلقائياً في مجلد الكاش
path = kagglehub.dataset_download("leftin/fruit-ripeness-unripe-ripe-and-rotten")
print(f"✅ مسار البيانات الأساسي: {path}")

# محاولة اكتشاف مجلد البيانات الصحيح (لأن أحياناً يكون داخل archive أو dataset)
possible_paths = [
    os.path.join(path, "fruit_ripeness_dataset", "archive (1)", "dataset"),
    os.path.join(path, "fruit_ripeness_dataset", "dataset"),
    os.path.join(path, "dataset"),
    path
]

BASE_DATA_DIR = None
for p in possible_paths:
    if os.path.exists(os.path.join(p, "train")):
        BASE_DATA_DIR = p
        break

if BASE_DATA_DIR is None:
    print("❌ خطأ: لم يتم العثور على مجلد 'train'. يرجى التأكد من هيكل البيانات.")
    raise SystemExit

TRAINING_DIR = os.path.join(BASE_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(BASE_DATA_DIR, "test")

print(f"📂 مسار التدريب: {TRAINING_DIR}")
print(f"📂 مسار الاختبار: {VALIDATION_DIR}")

# معلمات التدريب
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS     = 50 # يمكنك زيادتها إلى 50

tf.random.set_seed(42)

# ==========================
# 2) تجهيز البيانات
# ==========================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

print("⏳ جاري قراءة الصور...")

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# حفظ أسماء الكلاسات (مهم جداً للملفات الأخرى)
class_indices = train_generator.class_indices
NUM_CLASSES = len(class_indices)
print(f"✅ عدد الكلاسات: {NUM_CLASSES}")

with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(class_indices, f, ensure_ascii=False, indent=2)
print("✅ تم حفظ ملف class_indices.json")

# ==========================
# 3) بناء النموذج
# ==========================

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

# ==========================
# 4) التدريب والحفظ
# ==========================

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "thamara_ripeness_best.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

print("🚀 بدء التدريب...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint],
    verbose=1
)

model.save("thamara_ripeness_best.keras")
print("\n🎉 تم الانتهاء وحفظ النموذج!")
print("تم حفظ النموذج بنجاح.")
