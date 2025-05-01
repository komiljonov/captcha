import os
import requests
import base64
from PIL import Image, ImageTk
from io import BytesIO
import csv
import tkinter as tk
from tkinter import messagebox


# === Constants ===
URL = "https://127.0.0.1/ISAPI/Bumblebee/Platform/V0/VerificationCodeImage?CT=0&MT=GET"
HEADERS = {"Content-Type": "application/json"}
SAVE_DIR = "images"
LABEL_FILE = "labels.csv"
NUM_IMAGES = 50
VERIFY_SSL = False

# Ensure output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Tkinter app
root = tk.Tk()
root.title("Image Labeling Tool")
root.geometry("400x400")
root.resizable(False, False)

panel = tk.Label(root)
panel.pack(pady=20)

entry = tk.Entry(root, font=("Arial", 14))
entry.pack()

state_txt = tk.StringVar()
state_txt.set("Labeling in progress...")

state_label = tk.Label(root, textvariable=state_txt, font=("Arial", 12))
state_label.pack(pady=10)

# Variables
labels = []
images_to_label = []
current_image_idx = 0


# === Helper: Load unlabeled images from folder ===
def load_unlabeled_existing_images():
    labeled_files = set()
    if os.path.isfile(LABEL_FILE):
        with open(LABEL_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled_files.add(row["filename"])

    for file_name in sorted(os.listdir(SAVE_DIR)):
        if file_name.lower().endswith(".png") and file_name not in labeled_files:
            file_path = os.path.join(SAVE_DIR, file_name)
            images_to_label.append((file_name, file_path))


# === Helper: Fetch more images if needed ===
def fetch_images():
    global images_to_label
    to_fetch = NUM_IMAGES - len(images_to_label)
    for i in range(to_fetch):
        try:
            response = requests.post(
                URL,
                headers=HEADERS,
                verify=VERIFY_SSL,
                json={
                    "VerificationCodeImageRequest": {
                        "UserName": "admin",
                        "AccountType": 1,
                    }
                },
            )
            data = response.json()
            b64_data = data["ResponseStatus"]["Data"]["VerificationInfo"][
                "VerificationCodeImage"
            ]
            verification_mark = data["ResponseStatus"]["Data"]["VerificationInfo"][
                "VerificationMark"
            ]
            img_data = base64.b64decode(b64_data)
            image = Image.open(BytesIO(img_data))

            file_name = f"api_{i+1:03}_{verification_mark}.png"
            file_path = os.path.join(SAVE_DIR, file_name)
            image.save(file_path)

            images_to_label.append((file_name, file_path))

        except Exception as e:
            print(f"❌ Failed to fetch image {i+1}: {e}")


# === Show image in UI ===
def show_image(index):
    file_name, file_path = images_to_label[index]
    img = Image.open(file_path)
    img = img.resize((300, 100))
    img_tk = ImageTk.PhotoImage(img)
    panel.img_tk = img_tk
    panel.config(image=img_tk)
    entry.delete(0, tk.END)
    entry.focus()
    entry.select_range(0, tk.END)
    state_txt.set(f"Images: {index + 1}/{len(images_to_label)}")


# === Save label ===
def save_label(event=None):
    global current_image_idx

    label = entry.get().strip()
    if not label:
        messagebox.showerror("Error", "Please enter a label before continuing.")
        return

    file_name, _ = images_to_label[current_image_idx]
    labels.append([file_name, label])

    current_image_idx += 1
    if current_image_idx < len(images_to_label):
        show_image(current_image_idx)
    else:
        finish_labeling()


# === Finish labeling and save CSV ===
def finish_labeling():
    file_exists = os.path.isfile(LABEL_FILE)
    with open(LABEL_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "label"])
        writer.writerows(labels)

    messagebox.showinfo("Done", f"✅ {len(labels)} images labeled and saved!")
    root.quit()


# === Save and exit button ===
def save_and_exit():
    finish_labeling()


# === Buttons ===
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

save_btn = tk.Button(
    btn_frame, text="Save Label and Next", command=save_label, font=("Arial", 12)
)
save_btn.pack(side=tk.LEFT, padx=10)

exit_btn = tk.Button(
    btn_frame, text="Save and Exit", command=save_and_exit, font=("Arial", 12)
)
exit_btn.pack(side=tk.LEFT, padx=10)

root.bind("<Return>", save_label)

# === Start ===
load_unlabeled_existing_images()
# if len(images_to_label) < NUM_IMAGES:
#     fetch_images()

if images_to_label:
    show_image(current_image_idx)
    root.mainloop()
else:
    messagebox.showinfo("Info", "🎉 All images already labeled!")
