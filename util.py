from datetime import datetime
from ultralytics import YOLO
import cv2
import string


# ===== OCR MODEL  =====
ocr_model = YOLO("reader_ocr.pt")   

print("OCR model classes:")
#print(ocr_model.names)

# ===== CLASS NAME → ARABIC CHAR =====
CLASS_TO_ARABIC = {
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',

    'alif': 'ا',
    'baa': 'ب',
    'taa': 'ت',
    'thaa': 'ث',
    'jeem': 'ج',
    'haa': 'ح',
    'khaa': 'خ',
    'daal': 'د',
    'zaal': 'ذ',
    'raa': 'ر',
    'zay': 'ز',
    'seen': 'س',
    'sheen': 'ش',
    'saad': 'ص',
    'daad': 'ض',
    'Taa': 'ط',
    'Thaa': 'ظ',
    'ain': 'ع',
    'ghayn': 'غ',
    'faa': 'ف',
    'qaaf': 'ق',
    'kaaf': 'ك',
    'laam': 'ل',
    'meem': 'م',
    'noon': 'ن',
    'waw': 'و',
    'yaa': 'ي'
}


def read_license_plate(license_plate_crop):

    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None

    if len(license_plate_crop.shape) == 2:
        license_plate_crop = cv2.cvtColor(license_plate_crop, cv2.COLOR_GRAY2BGR)

    results = ocr_model(license_plate_crop)

    if len(results) == 0 or results[0].boxes is None:
        return None, None

    detections = results[0].boxes
    names = results[0].names

    chars = []

    for box in detections:
        cls_id = int(box.cls[0])
        raw_label = names[cls_id]
        label = CLASS_TO_ARABIC.get(raw_label, raw_label)
        conf = float(box.conf[0])

        if conf < 0.6:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2

        chars.append({
            "label": label,
            "conf": conf,
            "x": x_center
        })

    if len(chars) == 0:
        return None, None

    # ===== plate mid =====
    xs = [c["x"] for c in chars]
    plate_mid = (min(xs) + max(xs)) / 2

    arabic_part = []
    digit_part = []

    for c in chars:
        if c["x"] > plate_mid:
            arabic_part.append(c)
        else:
            digit_part.append(c)

    # sort spatially
    arabic_part = sorted(arabic_part, key=lambda x: x["x"], reverse=True)  # RTL
    digit_part = sorted(digit_part, key=lambda x: x["x"])                 # LTR

    arabic_text = " ".join([c["label"] for c in arabic_part])
    digit_text = "".join([c["label"] for c in digit_part])

    final_text = arabic_text + " " + digit_text
    avg_score = sum(c["conf"] for c in chars) / len(chars)

    return final_text, avg_score





def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score','date'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                                                            frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'],today_date )
                            )
        f.close()

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
