import ast
import cv2
import numpy as np
import pandas as pd
import csv
import os
import pygame  

# === Arabic RTL support ===
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display

# اسم ملف الصوت بتاعك
ALERT_SOUND = "assets_alarm.mp3"

# تهيئة pygame للصوت (مرة واحدة في البداية)
pygame.mixer.init()

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

def put_arabic_text(img, text, position, font_path="Amiri-Regular.ttf", font_size=90):
    display_text = str(text)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"خطأ في تحميل الخط: {e}")
        font = ImageFont.load_default()
    draw.text(position, display_text, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ────────────────────────────────────────────────
#           قراءة اللوحة المطلوبة من CSV
# ────────────────────────────────────────────────

wanted_plate = None
wanted_normalized = None

if os.path.exists('wanted_plates.csv'):
    with open('wanted_plates.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None)  
        row = next(reader, None)
        if row and row[0]:
            wanted_plate = row[0].strip()
            wanted_normalized = ''.join(wanted_plate.split())
            print(f"جاري البحث عن اللوحة المطلوبة: '{wanted_plate}'")
        else:
            print("ملف wanted_plates.csv فاضي")
else:
    print("ما فيش ملف wanted_plates.csv → ما فيش لوحة مطلوبة")

# عشان الصوت ما يتكررش لنفس السيارة
alert_played = {}  # car_id → True/False

# ────────────────────────────────────────────────
#                   Main Visualization
# ────────────────────────────────────────────────

results = pd.read_csv('test_interpolated.csv')

video_path = 'egypt_plate3.mp4'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    car_results = results[results['car_id'] == car_id]
    car_results = car_results[car_results['license_number'] != '0']
    
    if len(car_results) == 0:
        continue
    
    car_results['text_length'] = car_results['license_number'].apply(lambda x: len(x.replace(' ', '')))
    valid_results = car_results[car_results['text_length'] >= 4]
    
    if len(valid_results) > 0:
        valid_results['weighted_score'] = valid_results['license_number_score'] * valid_results['text_length']
        max_idx = valid_results['weighted_score'].idxmax()
        best_score = valid_results.loc[max_idx, 'license_number_score']
        best_number = valid_results.loc[max_idx, 'license_number']
        best_frame = valid_results.loc[max_idx, 'frame_nmr']
    else:
        max_ = np.amax(car_results['license_number_score'])
        best_score = max_
        best_number = car_results[car_results['license_number_score'] == max_]['license_number'].iloc[0]
        best_frame = car_results[car_results['license_number_score'] == max_]['frame_nmr'].iloc[0]
    
    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': best_number
    }
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
    ret, frame = cap.read()
    x1, y1, x2, y2 = ast.literal_eval(
        results[(results['car_id'] == car_id) &
                (results['license_number_score'] == best_score)]
        ['license_plate_bbox'].iloc[0]
        .replace('[ ', '[').replace('   ', ' ')
        .replace('  ', ' ').replace(' ', ',')
    )
    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(
        license_crop,
        (int((x2 - x1) * 400 / (y2 - y1)), 400)
    )
    license_plate[car_id]['license_crop'] = license_crop

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret = True

found = False  # عشان نعرف لو لقينا اللوحة ولا لأ

while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                df_.iloc[row_indx]['car_bbox']
                .replace('[ ', '[').replace('   ', ' ')
                .replace('  ', ' ').replace(' ', ',')
            )

            car_id = int(df_.iloc[row_indx]['car_id'])
            plate_text = df_.iloc[row_indx]['license_number']

            is_wanted = False

            if wanted_normalized and plate_text != '0':
                # تنظيف كامل للنصوص الاتنين
                input_clean = ''.join(wanted_plate.split()).replace(' ', '')
                plate_clean = ''.join(str(plate_text).split()).replace(' ', '')

                # تحويل الأرقام الشرقية للغربية في الاتنين عشان المقارنة تكون متساوية
                arabic_to_western = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
                input_clean = input_clean.translate(arabic_to_western)
                plate_clean = plate_clean.translate(arabic_to_western)

                # مقارنة تامة بعد التنظيف
                if input_clean == plate_clean:
                    is_wanted = True
                    found = True

            color = (0, 0, 255) if is_wanted else (0, 255, 0)
            thickness = 35 if is_wanted else 25

            draw_border(frame,
                        (int(car_x1), int(car_y1)),
                        (int(car_x2), int(car_y2)),
                        color=color, thickness=thickness)

            x1, y1, x2, y2 = ast.literal_eval(
                df_.iloc[row_indx]['license_plate_bbox']
                .replace('[ ', '[').replace('   ', ' ')
                .replace('  ', ' ').replace(' ', ',')
            )
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 0, 255), 12)

            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
            H, W, _ = license_crop.shape
            try:
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                frame = put_arabic_text(
                    frame,
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    (int((car_x2 + car_x1 - W) / 2), int(car_y1 - H - 300))
                )
            except Exception as e:
                print("DRAW ERROR:", e)

            # تشغيل الصوت بـ pygame مرة واحدة فقط لو لقينا اللوحة
            if is_wanted and not alert_played.get(car_id, False):
                try:
                    pygame.mixer.music.load(ALERT_SOUND)
                    pygame.mixer.music.play()
                    print(f"إنذار! تم العثور على اللوحة {wanted_plate} للسيارة {car_id}")
                except Exception as e:
                    print("مشكلة في تشغيل الصوت:", e)
                alert_played[car_id] = True

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

out.release()
cap.release()
cv2.destroyAllWindows()

# بعد ما الفيديو يخلّص، نطبع النتيجة النهائية
if wanted_plate:
    if found:
        print("\n✅ DONE FOUND  🚨")
    else:
        print("\n❌ Not found the wanted plate in the video.")
else:
    print("\n ⚠️ No wanted plate specified, so no search was performed.")