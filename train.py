import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import CaptchaModel
from dataset import CaptchaDataset
from utils import char_to_idx

# === Select CSV file with input ===
csv_options = ["train.csv", "test.csv", "labels.csv"]

print("📂 Select CSV file for training:")
for idx, fname in enumerate(csv_options, 1):
    print(f"{idx}. {fname}")

while True:
    try:
        selected_idx = int(input("Enter file number (1–3): "))
        selected_file = csv_options[selected_idx - 1]
        print(f"✅ Selected file: {selected_file}")
        break
    except (ValueError, IndexError):
        print("❌ Invalid selection. Please enter a number between 1 and 3.")

# === Settings ===
BATCH_SIZE = 512
IMG_WIDTH = 100
IMG_HEIGHT = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️ Using device: {DEVICE}")

# === Dataset & Dataloader ===
dataset = CaptchaDataset("images", selected_file, char_to_idx, IMG_WIDTH, IMG_HEIGHT)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x
)

# Use all CPU threads
torch.set_num_threads(os.cpu_count())
print(f"🧵 Using {torch.get_num_threads()} CPU threads")

# === Model, Loss, Optimizer ===
model = CaptchaModel(num_classes=len(char_to_idx) + 1)
model.to(DEVICE)
ctc_loss = nn.CTCLoss(blank=len(char_to_idx))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# === Event loop training ===
total_epoch_counter = 0

while True:
    try:
        epochs = int(
            input("🔁 How many epochs to train this round? (Enter 0 to exit): ")
        )
        if epochs <= 0:
            print("🛑 Training stopped by user.")
            break

        for i in range(epochs):
            model.train()
            total_loss = 0

            for batch in loader:
                imgs, labels = zip(*batch)
                imgs = torch.stack(imgs).to(DEVICE)

                labels_flat = torch.cat(labels).to(DEVICE)
                label_lengths = torch.tensor([len(label) for label in labels]).to(
                    DEVICE
                )

                preds = model(imgs).log_softmax(2)

                input_lengths = torch.full(
                    size=(preds.size(0),), fill_value=preds.size(1), dtype=torch.long
                ).to(DEVICE)

                loss = ctc_loss(
                    preds.permute(1, 0, 2), labels_flat, input_lengths, label_lengths
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            total_epoch_counter += 1
            avg_loss = total_loss / len(loader)
            print(f"📊 Epoch [{total_epoch_counter}], Loss: {avg_loss:.4f}")

        # === Save after this training round ===
        model_path = f"models/captchamodel_{total_epoch_counter}_loss{avg_loss:.4f}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}\n")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
        break
