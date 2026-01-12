import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import os
import json

# ==========================
# إعداد المسارات
# ==========================
MODEL_PATH = 'thamara_ripeness_best.keras'
INDICES_PATH = 'class_indices.json'
NEW_IMAGE_PATH = 'test_fruit.jpg'  # ضع صورة هنا لتجربتها
IMAGE_SIZE = (128, 128)

# 1. تحميل أسماء الكلاسات تلقائياً
if os.path.exists(INDICES_PATH):
    with open(INDICES_PATH, 'r', encoding='utf-8') as f:
        indices = json.load(f)
        # نحتاج عكس القاموس ليصبح {0: 'apple', 1: 'banana'}
        CLASS_NAMES = {v: k for k, v in indices.items()}
else:
    print(f"⚠️ تحذير: ملف {INDICES_PATH} غير موجود. تأكد من تشغيل ملف التدريب أولاً.")
    raise SystemExit

# 2. تحميل النموذج
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ تم تحميل النموذج بنجاح")
else:
    print("❌ خطأ: ملف النموذج غير موجود.")
    raise SystemExit

# ==========================
# دوال المساعدة
# ==========================
def decode_class(predicted_folder_name):
    # تحويل اسم المجلد (مثل freshapples) إلى كلمات عربية
    name_lower = predicted_folder_name.lower()
    
    # تحديد الفاكهة
    fruit_ar = "فاكهة غير معروفة"
    if "apple" in name_lower: fruit_ar = "تفاح"
    elif "banana" in name_lower: fruit_ar = "موز"
    elif "orange" in name_lower: fruit_ar = "برتقال"

    # تحديد الحالة
    ripeness_ar = "حالة غير معروفة"
    if "fresh" in name_lower: ripeness_ar = "طازجة"
    elif "rotten" in name_lower: ripeness_ar = "متعفنة"
    elif "unripe" in name_lower: ripeness_ar = "غير ناضجة"

    return ripeness_ar, fruit_ar

def predict_image(img_path):
    # تجهيز الصورة
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # التوقع
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index] * 100
    
    # الحصول على اسم المجلد الأصلي من الـ JSON
    folder_name = CLASS_NAMES[predicted_index]
    
    # الترجمة للعربية
    ripeness_ar, fruit_ar = decode_class(folder_name)

    print("\n--- 🍎 نتيجة نظام ثمرة 🍎 ---")
    print(f"الفاكهة: {fruit_ar}")
    print(f"الحالة:  {ripeness_ar}")
    print(f"الدقة:   {confidence:.2f}%")
    print(f"المجلد:  {folder_name}")
    print("-----------------------------\n")

if __name__ == '__main__':
    if os.path.exists(NEW_IMAGE_PATH):
        predict_image(NEW_IMAGE_PATH)
    else:
        print(f"ℹ️ نصيحة: ضع صورة باسم '{NEW_IMAGE_PATH}' بجانب الملف لتجربتها.")
        print(f"❌ لم يتم العثور على صورة الاختبار. يرجى وضع ملف {NEW_IMAGE_PATH} في مجلد المشروع.")
