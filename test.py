import torch
import torch.nn.functional as F
import cv2
import numpy as np
from model import CaptchaModel
from utils import char_to_idx, idx_to_char

# Settings
IMG_WIDTH = 100
IMG_HEIGHT = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of classes (+1 for blank token)
num_classes = len(char_to_idx) + 1

# Load model
model = CaptchaModel(num_classes=num_classes)
model.load_state_dict(torch.load("captcha_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W)
    img = np.expand_dims(img, axis=0)  # (1, 1, H, W)
    return torch.FloatTensor(img).to(DEVICE)


def decode_prediction(preds):
    preds = preds.softmax(2)  # softmax over classes
    preds = preds.argmax(2)  # take class with max probability
    preds = preds.squeeze(0)  # remove batch dimension

    # Collapse repeating characters and remove blanks
    prev_char = None
    output = ""
    for idx in preds:
        idx = idx.item()
        if idx != num_classes - 1:  # Ignore blank
            char = idx_to_char[idx]
            if char != prev_char:
                output += char
                prev_char = char
        else:
            prev_char = None
    return output


def predict(img_path):
    img = preprocess_image(img_path)
    with torch.no_grad():
        preds = model(img)
    text = decode_prediction(preds)
    return text


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image")
        sys.exit(1)

    img_path = sys.argv[1]
    prediction = predict(img_path)
    print(f"Predicted text: {prediction}")
