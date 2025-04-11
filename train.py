import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import time
from tokenizer import SimpleTokenizer
from load_data import load_dataset
from model import Encoder, AttentionMechanism, Decoder, Seq2SeqModel

# Config
EMBED_SIZE = 64
HIDDEN_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 2
LEARNING_RATE = 0.0005
MODEL_NAME = "chatbot_model.pt"
DATASET_FILE = "processed_dailydialog/dailydialog_train_input_response.txt"
MODEL_DIR = "models"
VOCAB_FILE = "processed_dailydialog/vocab.txt"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("Loading data...")
input_texts, target_texts = load_dataset(DATASET_FILE)
tokenizer = SimpleTokenizer()
START_TOKEN, END_TOKEN, PAD_TOKEN = "<start>", "<end>", "<pad>"
input_texts = [t.lower() for t in input_texts]
target_texts = [f"{START_TOKEN} {t.lower()} {END_TOKEN}" for t in target_texts]

tokenizer.build_vocab(input_texts + target_texts)
tokenizer.save_vocab(VOCAB_FILE)

vocab_size = len(tokenizer.word2idx)

input_seqs = [torch.tensor(tokenizer.encode(t)) for t in input_texts]
target_seqs = [torch.tensor(tokenizer.encode(t)) for t in target_texts]
split = int(0.9 * len(input_seqs))
train_input, val_input = input_seqs[:split], input_seqs[split:]
train_target, val_target = target_seqs[:split], target_seqs[split:]

combined = list(zip(train_input, train_target))
random.shuffle(combined)
train_input[:], train_target[:] = zip(*combined)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE, dropout_rate=0.3)
attn = AttentionMechanism(HIDDEN_SIZE)
decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE, attn, dropout_rate=0.3, pad_idx=tokenizer.word2idx[PAD_TOKEN])
model = Seq2SeqModel(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx[PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    start_time = time.time()  # Start timer
    model.train()
    total_loss = 0

    for i in range(0, len(train_input), BATCH_SIZE):
        inputs = torch.nn.utils.rnn.pad_sequence(train_input[i:i+BATCH_SIZE], batch_first=True,
                                                 padding_value=tokenizer.word2idx[PAD_TOKEN]).to(device)
        targets = torch.nn.utils.rnn.pad_sequence(train_target[i:i+BATCH_SIZE], batch_first=True,
                                                  padding_value=tokenizer.word2idx[PAD_TOKEN]).to(device)
        optimizer.zero_grad()
        output = model(inputs, targets)
        loss = criterion(output[:, 1:].reshape(-1, vocab_size), targets[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_input)

    # === Validation ===
    model.eval()
    val_loss, correct_tokens, total_tokens = 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(val_input), BATCH_SIZE):
            inputs = torch.nn.utils.rnn.pad_sequence(val_input[i:i+BATCH_SIZE], batch_first=True,
                                                     padding_value=tokenizer.word2idx[PAD_TOKEN]).to(device)
            targets = torch.nn.utils.rnn.pad_sequence(val_target[i:i+BATCH_SIZE], batch_first=True,
                                                      padding_value=tokenizer.word2idx[PAD_TOKEN]).to(device)
            output = model(inputs, targets, teacher_forcing_ratio=0.0)
            preds = output.argmax(dim=2)

            mask = targets != tokenizer.word2idx[PAD_TOKEN]
            correct_tokens += (preds == targets).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            val_loss += criterion(output[:, 1:].reshape(-1, vocab_size), targets[:, 1:].reshape(-1)).item()

    avg_val_loss = val_loss / len(val_input)
    val_acc = 100.0 * correct_tokens / total_tokens if total_tokens > 0 else 0.0

    end_time = time.time()  # End timer
    epoch_time = end_time - start_time  # Calculate elapsed time

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f} seconds")

# Save model
final_model_path = os.path.join(MODEL_DIR, MODEL_NAME)
torch.save(model.state_dict(), final_model_path)
print("Model saved to:", final_model_path)