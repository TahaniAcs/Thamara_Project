import kagglehub
import os

DATASET_ID = "leftin/fruit-ripeness-unripe-ripe-and-rotten"
print(f"بدء تحميل مجموعة البيانات: {DATASET_ID}")

# تحميل أحدث إصدار من مجموعة البيانات
# سيقوم kagglehub بتنزيل الملفات وفك ضغطها
path = kagglehub.dataset_download(DATASET_ID)

print("\n✅ تم التحميل بنجاح!")
print("مسار ملفات مجموعة البيانات:", path)