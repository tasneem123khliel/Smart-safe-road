# 🚗 Egyptian License Plate Recognition & Stolen Car Detection

> Graduation Project — Automatic detection and reading of Egyptian license plates from video footage, with real-time stolen/wanted car alert system.

---

## 📸 Demo Output

| Frame 1 | Frame 2 | Frame 3 |
|---------|---------|---------|
| ![out1](output/out1.PNG) | ![out2](output/out2.PNG) | ![out3](output/out3.PNG) |

> The system detects vehicles, reads Arabic license plates (e.g. `انل ٣٥١٣`, `ط ف س ٥٨٦`, `س ل ج ٨٢٢١`), and highlights them in real time.

---

## 📌 Project Overview

This system processes traffic video footage to:

- **Detect vehicles** using YOLOv8
- **Detect and crop license plates** using a custom-trained YOLO model
- **Read Arabic license plate text** using a custom OCR model
- **Track vehicles** across frames using SORT (Simple Online and Realtime Tracking)
- **Interpolate missing detections** to fill gaps between frames
- **Alert in real time** when a wanted/stolen plate is found (sound + visual highlight)

---

## 🗂️ Project Structure

```
├── main.py                  # Main pipeline: detection, tracking, OCR, CSV output
├── util.py                  # OCR reading logic, CSV writing, car matching
├── add_missing_data.py      # Interpolates missing bounding boxes between frames
├── visualize.py             # Renders annotated output video with Arabic text & alerts
├── wanted_plates.csv        # Auto-generated file storing the searched plate
├── tests.csv                # Raw detection results (output of main.py)
├── test_interpolated.csv    # Interpolated results (output of add_missing_data.py)
├── out.mp4                  # Final annotated output video
├── egypt_plate3.mp4         # Input video
├── yolov8n.pt               # YOLOv8 vehicle detection model
├── plate_detector.pt        # Custom YOLO license plate detector
├── reader_ocr.pt            # Custom YOLO-based Arabic OCR model
└── Amiri-Regular.ttf        # Arabic font for on-video text rendering
```

---

## ⚙️ How It Works

### Step 1 — Detection & OCR (`main.py`)

1. User enters a wanted plate number (or leaves blank for no search)
2. YOLOv8 detects vehicles (cars, buses, motorcycles, trucks) in each frame
3. SORT tracker assigns consistent IDs to vehicles across frames
4. Custom YOLO model detects license plates within vehicle bounding boxes
5. Arabic OCR model reads the plate characters
6. Results saved to `tests.csv` with frame number, car ID, bounding boxes, plate text, and date

### Step 2 — Interpolation (`add_missing_data.py`)

- Reads `tests.csv` and fills in missing frames per vehicle using linear interpolation
- Preserves original plate readings; marks imputed frames with score `0`
- Outputs `test_interpolated.csv`

### Step 3 — Visualization (`visualize.py`)

- Reads `test_interpolated.csv` and the original video
- Selects the best plate reading per vehicle (highest confidence × text length)
- Renders the annotated video with:
  - Green border around tracked vehicles
  - Red rectangle around the license plate
  - Cropped plate image shown above the vehicle
  - Arabic plate number displayed in Amiri font
  - **Red border + alarm sound** if the plate matches the wanted plate

---

## 🔠 Arabic OCR Details

The OCR model (`reader_ocr.pt`) detects individual Arabic characters and digits. Each detection is mapped to its Arabic Unicode character:

- Arabic letters: `alif → ا`, `baa → ب`, `taa → ت` ... etc.
- Arabic-Indic digits: `0 → ٠`, `1 → ١` ... `9 → ٩`

**Plate layout logic:**
- Characters to the **right** of the plate midpoint → Arabic letters (sorted RTL)
- Characters to the **left** → digits (sorted LTR)
- Final format: `[letters] [digits]` (e.g. `انل ٣٥١٣`)

Only detections with confidence ≥ 0.6 are kept.

---

## 🚨 Stolen Car Alert

When the user enters a plate number at startup:

```
Enter Your plate?: انل ٣٥١٣
```

- The plate is saved to `wanted_plates.csv`
- During visualization, every detected plate is compared (after normalizing spaces and converting Arabic-Indic digits to Western digits)
- If a match is found:
  - Vehicle border turns **red**
  - An **alarm sound** (`assets_alarm.mp3`) plays once per vehicle
  - A `✅ DONE FOUND 🚨` message is printed at the end

---

## 🛠️ Requirements

```bash
pip install ultralytics opencv-python numpy scipy pandas pygame pillow arabic-reshaper python-bidi
```

Also required:
- [`sort`](https://github.com/abewley/sort) — place in `sort/` directory
- Custom model files: `plate_detector.pt`, `reader_ocr.pt`
- Font file: `Amiri-Regular.ttf`
- Alarm audio: `assets_alarm.mp3`

---

## ▶️ Usage

```bash
# Step 1: Run detection on video
python main.py

# Step 2: Fill in missing frames
python add_missing_data.py

# Step 3: Generate annotated output video
python visualize.py
```

---

## 📁 Full Project

The complete project (including models, video, and assets) is available on Google Drive:

🔗 [Google Drive — Full Project](https://drive.google.com/drive/folders/163XeFDXoMJeZw_nac58bN8pon2_oiXsG?usp=sharing)

---

## 📅 Date Tracking

Every detection record is automatically stamped with the current date (`YYYY-MM-DD`) and saved to CSV — useful for building a historical log of plate sightings.

---

## 📝 Notes

- The system is designed for **Egyptian license plates** which use Arabic letters and Arabic-Indic numerals
- Works on rear-view traffic footage
- Interpolation improves visualization smoothness when plates are temporarily occluded
- The wanted plate comparison is **normalization-aware**: spaces and digit script differences are handled automatically

  ## 👩‍💻 Author

**Tasneem Yasser**
- Graduation Project — Egyptian License Plate Recognition & Stolen Car Detection
