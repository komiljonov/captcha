import os
import requests
import base64
from PIL import Image
from io import BytesIO
import csv

# Constants
URL = "https://127.0.0.1/ISAPI/Bumblebee/Platform/V0/VerificationCodeImage?CT=0&MT=GET"
HEADERS = {"Content-Type": "application/json"}
SAVE_DIR = "images"
LABEL_FILE = "labels.csv"
NUM_IMAGES = 10
VERIFY_SSL = False  # Change to True if you have a valid cert

# Ensure output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Label storage
labels = []

# Check if label file exists
file_exists = os.path.isfile(LABEL_FILE)

for i in range(NUM_IMAGES):
    print(f"[{i+1}/{NUM_IMAGES}] Requesting image...")

    try:
        response = requests.post(
            URL,
            headers=HEADERS,
            verify=VERIFY_SSL,
            json={
                "VerificationCodeImageRequest": {"UserName": "admin", "AccountType": 1}
            },
        )
        data = response.json()

        print(data)

        b64_data = data["ResponseStatus"]["Data"]["VerificationInfo"][
            "VerificationCodeImage"
        ]
        verification_mark = data["ResponseStatus"]["Data"]["VerificationInfo"][
            "VerificationMark"
        ]

        # Decode base64 and save image
        img_data = base64.b64decode(b64_data)
        image = Image.open(BytesIO(img_data))
        file_name = f"{i+1:03}_{verification_mark}.png"
        file_path = os.path.join(SAVE_DIR, file_name)
        image.save(file_path)
        print(f"Saved: {file_path}")

        # Manual labeling
        image.show()
        label = input(f"Enter text for image {file_name}: ").strip()
        labels.append([file_name, label])
        image.close()

    except Exception as e:
        print(f"Failed to fetch/save/label image {i+1}: {e}")

# Save new labels to CSV without overwriting old ones
with open(LABEL_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["filename", "label"])  # Only write header once
    writer.writerows(labels)

print(f"Done! {len(labels)} new images labeled and appended.")
