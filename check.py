import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd

from model import CaptchaModel
from utils import char_to_idx, idx_to_char

# === Settings ===
IMG_WIDTH = 100
IMG_HEIGHT = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "images"
MODEL_DIR = "models"
num_classes = len(char_to_idx) + 1  # +1 for CTC blank token

# === Select LABEL_FILE ===
csv_choices = ["train.csv", "test.csv", "labels.csv","news.csv"]
print("📂 Select label file:")
for i, name in enumerate(csv_choices, 1):
    print(f"{i}. {name}")

while True:
    try:
        idx = int(input("Enter number (1-3): "))
        LABEL_FILE = csv_choices[idx - 1]
        print(f"✅ Selected: {LABEL_FILE}")
        break
    except (ValueError, IndexError):
        print("❌ Invalid selection, try again.")


# === Model loading ===
def load_model_interactively():
    available_models = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")])
    if not available_models:
        print("❌ No model files found in 'models/' directory.")
        exit()

    print("\n📦 Available models:")
    for i, name in enumerate(available_models, 1):
        print(f"{i}. {name}")

    while True:
        try:
            idx = int(input("Select model number: "))
            selected_model = available_models[idx - 1]
            break
        except (ValueError, IndexError):
            print("❌ Invalid selection, try again.")

    model_path = os.path.join(MODEL_DIR, selected_model)
    model = CaptchaModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Loaded model: {selected_model}")
    return model


# === Preprocessing ===
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W)
    img = np.expand_dims(img, axis=0)  # (1, 1, H, W)
    return torch.FloatTensor(img).to(DEVICE)


# === Decoder ===
def decode_prediction(preds):
    preds = preds.softmax(2)  # softmax over classes
    preds = preds.argmax(2)  # max probability index
    preds = preds.squeeze(0)  # remove batch dim

    output = ""
    prev_char = None
    for idx in preds:
        idx = idx.item()
        if idx != num_classes - 1:  # ignore CTC blank
            char = idx_to_char[idx]
            if char != prev_char:
                output += char
                prev_char = char
        else:
            prev_char = None
    return output


# === Predict one image ===
def predict(model, img_path):
    img = preprocess_image(img_path)
    with torch.no_grad():
        preds = model(img)
    return decode_prediction(preds)


# === Main evaluation loop ===
if __name__ == "__main__":
    model = load_model_interactively()

    df = pd.read_csv(LABEL_FILE)
    total = 0
    correct = 0
    wrong_samples = []

    for _, row in df.iterrows():
        filename = row["filename"]
        true_label = row["label"]
        img_path = os.path.join(IMAGE_DIR, filename)

        try:
            pred_label = predict(model, img_path)
            is_correct = pred_label == true_label
            total += 1
            correct += int(is_correct)

            if not is_correct:
                wrong_samples.append((filename, true_label, pred_label))

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    if wrong_samples:
        print("\n❌ Wrong predictions:")
        for filename, true_label, pred_label in wrong_samples:
            print(f"images/{filename}: True='{true_label}' | Predicted='{pred_label}'")
    else:
        print("\n🎯 All predictions correct!")

    accuracy = correct / total * 100 if total else 0
    print(f"\n✅ Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")
