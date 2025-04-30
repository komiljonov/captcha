import pandas as pd
import questionary

# === Load full dataset ===
csv_file = "labels.csv"  # Your original dataset
df = pd.read_csv(csv_file)
total = len(df)

# === Ask percentage for test split ===
percent = questionary.text(
    "🔢 What percent of data should be used for testing? (e.g., 20)",
    validate=lambda text: text.isdigit() and 0 < int(text) < 100,
).ask()

test_percent = int(percent)
test_size = int(total * test_percent / 100)
train_size = total - test_size

# === Shuffle and split ===
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_test = df_shuffled[:test_size]
df_train = df_shuffled[test_size:]

# === Save splits ===
df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)

print(f"\n✅ Split complete:")
print(f"🟢 train.csv → {train_size} samples")
print(f"🔵 test.csv  → {test_size} samples ({test_percent}%)")
