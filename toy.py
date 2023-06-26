from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Change num_labels as per your classification task

# Example text classification data
texts = ['This is a ,''positive sentence.', 'This is a negative sentence.']
labels = [1, 0]

# Split the data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize the texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create PyTorch datasets
class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextClassificationDataset(train_encodings, train_labels)
val_dataset = TextClassificationDataset(val_encodings, val_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(3):  # Adjust the number of epochs as needed
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    preds = []
    true_labels = []
    for batch in val_loader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(inputs['labels'].tolist())

    val_accuracy = accuracy_score(true_labels, preds)

    print(f"Epoch: {epoch+1}")
    print(f"Training Loss: {avg_train_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    print("---------------------------")



# Estimate labels for a new text
new_text = ['I love you so much', 'you are good', 'so sexy']

# Tokenize the new text
new_text_encodings = tokenizer(new_text, truncation=True, padding=True)

# Create a dataset for the new text
new_text_dataset = TextClassificationDataset(new_text_encodings, [0]* len(new_text))  # Provide a dummy label since it's not used in inference

# Create a data loader for the new text
new_text_loader = DataLoader(new_text_dataset, batch_size=1, shuffle=False)

# Switch model to evaluation mode
model.eval()

for batch in new_text_loader:
    inputs = {key: val.to(device) for key, val in batch.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    # Print the estimated label
    estimated_label = "Positive" if predicted_label == 1 else "Negative"
    print(f"Estimated Label for '{new_text[0]}': {estimated_label}")