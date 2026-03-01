import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ========================
# CONFIG
# ========================
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 2
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# LOAD DATA
# ========================
print("Loading WELFake (train)...")
wel = pd.read_csv("data/wel_fake/WELFake_Dataset.csv")
wel["text"] = wel["title"].fillna("") + " " + wel["text"].fillna("")
wel = wel[["text", "label"]]

print("Loading ISOT (test)...")
fake_isot = pd.read_csv("data/isot/Fake.csv")
true_isot = pd.read_csv("data/isot/True.csv")

fake_isot["label"] = 1
true_isot["label"] = 0

isot = pd.concat([fake_isot, true_isot])
isot = isot[["text", "label"]]

print("Train size:", len(wel))
print("Test size:", len(isot))

# ========================
# TOKENIZER
# ========================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(wel["text"], wel["label"])
test_dataset = FakeNewsDataset(isot["text"], isot["label"])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ========================
# MODEL
# ========================
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# ========================
# TRAIN LOOP
# ========================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print("Training Loss:", total_loss / len(train_loader))

    # Evaluation
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)

    print("Test Accuracy:", acc)
    print("Test F1 Score:", f1)

# ========================
# SAVE PROPERLY
# ========================
print("Saving model and tokenizer...")

os.makedirs("saved_model", exist_ok=True)

model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

print("Everything saved inside /saved_model")