import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 1. تحديد معلمات النموذج والملفات

MODEL_PATH = 'thamara_ripeness_cnn_model.h5' # تأكد من أن هذا يتطابق مع اسم ملف النموذج المحفوظ
IMAGE_SIZE = (128, 128)
# تحديد الفئات (يجب أن تتطابق مع ترتيب الفئات أثناء التدريب)
CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshoranges', 
    'rottenapples', 'rottenbanana', 'rottenoranges', 
    'unripe apple', 'unripe banana', 'unripe orange'
]
# مسار الصورة الجديدة التي تريد اختبارها (استبدل باسم صورتك)
NEW_IMAGE_PATH = 'test_fruit.jpg/test2_orange.jpg' 

# 2. تحميل النموذج المدرب مسبقاً

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ تم تحميل النموذج بنجاح من: {MODEL_PATH}")
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج. تأكد أن ملف {MODEL_PATH} موجود: {e}")
    exit()

# 3. دالة تجهيز الصورة للتصنيف

def prepare_image(img_path):
    # تحميل الصورة وتغيير حجمها إلى الحجم الذي دربت عليه النموذج (128x128)
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    # تحويل الصورة إلى مصفوفة (Array)
    img_array = image.img_to_array(img)
    # إضافة بُعد batch (يجب أن يكون شكل المصفوفة (1, 128, 128, 3))
    img_array = np.expand_dims(img_array, axis=0)
    # تطبيع قيمة البكسلات (مثلما فعلنا في التدريب)
    img_array /= 255.0
    return img_array

# 4. دالة التنبؤ (Prediction)

def predict_image(model, img_path, class_names):
    # تجهيز الصورة
    processed_image = prepare_image(img_path)
    
    # إجراء التنبؤ
    predictions = model.predict(processed_image)
    
    # الحصول على الفئة ذات الاحتمالية الأعلى
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    
    # تحليل حالة النضج
    
    if "fresh" in predicted_class or "ripe" in predicted_class:
        status = "ناضجة - جاهزة للحصاد"
    elif "rotten" in predicted_class:
        status = "متعفنة - يجب التخلص منها"
    elif "unripe" in predicted_class:
        status = "غير ناضجة - انتظر"
    else:
        status = "تصنيف غير معروف"

    print("\n--- نتيجة نظام ثمرة ---")
    print(f"صنف الفاكهة: {predicted_class.replace('fresh', 'طازج').replace('rotten', 'متعفن').replace('unripe', 'غير ناضج')}")
    print(f"حالة النضج: {status}")
    print(f"درجة الثقة: {confidence:.2f}%")
    print("-------------------------\n")


# 5. تشغيل البرنامج

if __name__ == '__main__':
    if os.path.exists(NEW_IMAGE_PATH):
        predict_image(model, NEW_IMAGE_PATH, CLASS_NAMES)
    else:
        print(f"❌ لم يتم العثور على صورة الاختبار. يرجى وضع ملف {NEW_IMAGE_PATH} في مجلد المشروع.")