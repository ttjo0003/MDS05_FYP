# Sign Language Recognition Pipeline

This module implements a real-time sign language recognition system using MediaPipe Holistic and an LSTM model.

## Features
- Real-time webcam input
- Landmark extraction (pose + hands)
- Sequence-based prediction (30 frames)
- Frontend + backend integration

---

## 📁 Project Structure
```
sign_language_pipeline/
│
├── frontend/
│ ├── index.html # UI interface
│ └── app.js # Webcam + landmark extraction + API call
│
├── backend/
│ └── app.py # Flask inference server
│
├── model/
│ ├── best_sign_lstm.pth # Trained model weights
│ └── label_map.json # Label mapping
│
├── training/
│ ├── train_sign_lstm.ipynb # Model training notebook
│ └── wlasl_preprocess.py # Data preprocessing script
│
├── data/
│ ├── processed_holistic_pose_hand.zip
│ └── metadata_holistic_pose_hand.csv
│
└── README.md
```


---

## ⚙️ Requirements

Install required Python packages:

```bash
pip install flask flask-cors torch numpy

## 🚀 How to Run

### 1. Start Backend (Flask)

```bash
cd backend
python app.py
```

You should see:

```
Running on http://127.0.0.1:5000
```

---

### 2. Start Frontend

```bash
cd frontend
python -m http.server 8000
```

Open browser:

http://127.0.0.1:8000

---

### 3. Using the System

- Click **Start Prediction**
- Perform a sign gesture in front of webcam
- System collects 30 frames
- Prediction result will be displayed

---

## 🧠 Model Details

- Model: LSTM  
- Input size: 225 features per frame  
  - Pose: 33 × 3  
  - Left hand: 21 × 3  
  - Right hand: 21 × 3  
- Sequence length: 30 frames  
- Output: Sign label + confidence score  

---
## Dataset

The processed dataset is not included in this repository due to GitHub file size limitations.

### Download

You can download the dataset from the following link:
👉 [Google Drive Link](https://drive.google.com/file/d/1IRfsOEYGNzxO4PXDTxMRqQqY-sKmn6hu/view?usp=sharing)

### Setup

After downloading, place the `.zip` file in the following directory:

`sign_language_pipeline/data/`


Then extract it:

`unzip processed_holistic_pose_hand.zip`


### Alternative (Reproducibility)

You can also regenerate the dataset by running:

```bash
python training/wlasl_preprocess.py
```

### Metadata

`metadata_holistic_pose_hand.csv` contains:

- video_id  
- gloss (label)  
- split (train/val/test)  
- file path  

---

## 🔄 Data Preprocessing

Run:

```bash
python wlasl_preprocess.py
```

This will:

- Extract landmarks using MediaPipe Holistic  
- Convert videos into `.npy` feature sequences  
- Generate metadata CSV  

---

## 🏋️ Model Training

Open:

```
train_sign_lstm.ipynb
```

Steps:

1. Load processed dataset  
2. Train LSTM model  
3. Save model:

```python
torch.save(model.state_dict(), "best_sign_lstm.pth")
```

4. Save label mapping (`label_map.json`)

---

## 🌐 System Pipeline

```
Webcam → MediaPipe Holistic → 225-dim features
→ 30-frame sequence → Flask backend → LSTM model
→ Prediction → Frontend display
```