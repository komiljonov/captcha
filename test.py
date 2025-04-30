import os
import requests
import base64
import torch
import numpy as np
from io import BytesIO
from PIL import Image, ImageTk
import tkinter as tk
from model import CaptchaModel
from utils import char_to_idx, idx_to_char

# === Constants ===
URL = "https://127.0.0.1/ISAPI/Bumblebee/Platform/V0/VerificationCodeImage?CT=0&MT=GET"
HEADERS = {"Content-Type": "application/json"}
VERIFY_SSL = False
IMG_WIDTH = 100
IMG_HEIGHT = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "images"
NUM_IMAGES = 9
num_classes = len(char_to_idx) + 1

# Ensure images folder exists
os.makedirs(SAVE_DIR, exist_ok=True)


# === Choose model ===
def choose_model():
    models = sorted([f for f in os.listdir("models") if f.endswith(".pth")])
    if not models:
        print("❌ No models found!")
        exit()

    print("\n📦 Available models:")
    for i, name in enumerate(models, 1):
        print(f"{i}. {name}")

    while True:
        try:
            idx = int(input("Select model number: "))
            selected = models[idx - 1]
            break
        except (ValueError, IndexError):
            print("❌ Invalid selection.")

    path = os.path.join("models", selected)
    model = CaptchaModel(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Loaded: {selected}")
    return model


# === Fetch one image ===
def fetch_image():
    res = requests.post(
        URL,
        headers=HEADERS,
        verify=VERIFY_SSL,
        json={"VerificationCodeImageRequest": {"UserName": "admin", "AccountType": 1}},
    )
    data = res.json()
    b64 = data["ResponseStatus"]["Data"]["VerificationInfo"]["VerificationCodeImage"]
    return Image.open(BytesIO(base64.b64decode(b64)))


# === Predict ===
def preprocess(img: Image.Image):
    img = img.convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.FloatTensor(img).to(DEVICE)


def decode_prediction(preds):
    preds = preds.softmax(2).argmax(2).squeeze(0)
    output = ""
    prev_char = None
    for idx in preds:
        idx = idx.item()
        if idx != num_classes - 1:
            char = idx_to_char[idx]
            if char != prev_char:
                output += char
                prev_char = char
        else:
            prev_char = None
    return output


def predict(model, pil_img):
    x = preprocess(pil_img)
    with torch.no_grad():
        preds = model(x)
    return decode_prediction(preds)


# === UI ===
class App:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.images_data = []  # (PIL, tk_img, pred)

        self.frame = tk.Frame(root)
        self.frame.pack()

        self.refresh_button = tk.Button(
            root,
            text="🔄 Refresh Images",
            font=("Arial", 14),
            command=self.refresh_images,
        )
        self.refresh_button.pack(pady=10)

        self.refresh_images()

    def refresh_images(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

        self.images_data.clear()
        for i in range(NUM_IMAGES):
            pil_img = fetch_image()
            pred = predict(self.model, pil_img)
            print(f"🔍 [{i+1}] Prediction: {pred}")

            display_img = pil_img.resize((120, 40))
            tk_img = ImageTk.PhotoImage(display_img)
            self.images_data.append((pil_img, tk_img, pred))

        self.render_images()

    def render_images(self):
        for i, (pil_img, tk_img, pred) in enumerate(self.images_data):
            row, col = divmod(i, 3)
            index = row * 3 + col

            img_label = tk.Label(self.frame, image=tk_img)
            img_label.grid(row=row * 3, column=col, padx=10, pady=5)
            img_label.image = tk_img

            text = tk.Label(self.frame, text=f"📜 {pred}", font=("Arial", 12))
            text.grid(row=row * 3 + 1, column=col)

            def save_callback(pil_img=pil_img, pred=pred, index=index):
                filename = f"batch_{index+1}_{pred}.png"
                pil_img.save(os.path.join(SAVE_DIR, filename))
                print(f"💾 Saved: {filename}")

            save_btn = tk.Button(self.frame, text="💾 Save", command=save_callback)
            save_btn.grid(row=row * 3 + 2, column=col, pady=2)


# === Main ===
if __name__ == "__main__":
    model = choose_model()
    root = tk.Tk()
    root.title("CAPTCHA AI Viewer")
    app = App(root, model)
    root.mainloop()
