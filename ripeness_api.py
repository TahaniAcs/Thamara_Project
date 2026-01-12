import os
import json
import numpy as np
import tensorflow as tf
from io import BytesIO
from keras.utils import load_img, img_to_array
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================
# 1) إعداد النموذج والكلاسات
# ==========================

MODEL_PATH = "thamara_ripeness_best.keras" # مسار نسبي
INDICES_PATH = "class_indices.json"
IMAGE_SIZE = (128, 128)

# تحميل النموذج
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ لم يتم العثور على النموذج: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully")

# تحميل أسماء الكلاسات
CLASS_NAMES_MAP = {}
if os.path.exists(INDICES_PATH):
    with open(INDICES_PATH, 'r', encoding='utf-8') as f:
        indices = json.load(f)
        CLASS_NAMES_MAP = {v: k for k, v in indices.items()}
else:
    print("⚠️ تحذير: ملف class_indices.json غير موجود. قد تكون النتائج غير دقيقة.")
    # قائمة احتياطية في حال عدم وجود الملف (يجب أن تطابق ترتيب التدريب)
    # ملاحظة: هذه القائمة قد تكون خاطئة إذا تغير ترتيب المجلدات، لذا يفضل وجود ملف JSON
    pass 

# ==========================
# 2) إعداد FastAPI
# ==========================
app = FastAPI(title="Thamara Ripeness API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RipenessResponse(BaseModel):
    predicted_class_en: str
    predicted_class_ar: str
    ripeness_status_ar: str
    ripeness_status_en: str
    confidence: float
    score: float

# ==========================
# 3) دوال مساعدة
# ==========================
def prepare_image_bytes(file_bytes: bytes) -> np.ndarray:
    img = load_img(BytesIO(file_bytes), target_size=IMAGE_SIZE, color_mode="rgb")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def interpret_result(folder_name: str):
    name = folder_name.lower()
    
    # استخراج الفاكهة
    fruit_ar = "فاكهة"
    fruit_en = "Fruit"
    if "apple" in name: fruit_ar, fruit_en = "تفاح", "Apple"
    elif "banana" in name: fruit_ar, fruit_en = "موز", "Banana"
    elif "orange" in name: fruit_ar, fruit_en = "برتقال", "Orange"

    # استخراج الحالة
    status_ar, status_en, base_score = "غير معروف", "Unknown", 50
    if "fresh" in name:
        status_ar = "طازجة - جاهزة للحصاد"
        status_en = "Fresh - Ready to harvest"
        base_score = 90
    elif "rotten" in name:
        status_ar = "متعفنة - تخلص منها"
        status_en = "Rotten - Discard"
        base_score = 10
    elif "unripe" in name:
        status_ar = "غير ناضجة - انتظر"
        status_en = "Unripe - Wait"
        base_score = 40

    full_ar = f"{fruit_ar} {status_ar.split(' - ')[0]}" # تفاح طازجة
    return full_ar, status_ar, status_en, base_score

# ==========================
# 4) نقطة النهاية (Endpoint)
# ==========================
@app.post("/analyze", response_model=RipenessResponse)
async def analyze(image_file: UploadFile = File(...)):
    file_bytes = await image_file.read()
    img_tensor = prepare_image_bytes(file_bytes)
    
    preds = model.predict(img_tensor)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx]) * 100
    
    # الحصول على الاسم الحقيقي للمجلد
    predicted_folder = CLASS_NAMES_MAP.get(idx, "unknown")
    
    class_ar, status_ar, status_en, base_score = interpret_result(predicted_folder)
    
    # حساب سكور تقريبي للجودة
    final_score = base_score
    if "fresh" in predicted_folder.lower():
        final_score = min(100, base_score + (conf - 50) * 0.1)

    return RipenessResponse(
        predicted_class_en=predicted_folder,
        predicted_class_ar=class_ar,
        ripeness_status_ar=status_ar,
        ripeness_status_en=status_en,
        confidence=round(conf, 2),
        score=round(final_score, 1)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)