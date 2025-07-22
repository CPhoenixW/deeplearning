import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.parallel
import pickle
from tqdm import tqdm
import os
import time

# Configuration
INPUT_DIM = None
OUTPUT_DIM = None
HID_DIM = 256
EMB_DIM = 200
BATCH_SIZE = 100
MAX_LEN = 50
N_EPOCHS = 150
NUM_LAYERS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_PATH = "src/frac14/"

# Data Handling
def load_vocabs(zh_path, en_path):
    with open(zh_path, "rb") as f:
        zh_vocab = pickle.load(f)
    with open(en_path, "rb") as f:
        en_vocab = pickle.load(f)
    return zh_vocab, en_vocab

def load_encoded_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

class EncodedDataset(Dataset):
    def __init__(self, encoded):
        self.encoder_input = encoded["encoder_input"]
        self.decoder_input = encoded["decoder_input"]
        self.decoder_target = encoded["decoder_target"]

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoder_input[idx]),
            torch.tensor(self.decoder_input[idx]),
            torch.tensor(self.decoder_target[idx])
        )

# Model Definition
class Seq2SeqGRU(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, num_layers=2):
        super().__init__()
        self.embedding_enc = nn.Embedding(input_dim, emb_dim)
        self.embedding_dec = nn.Embedding(output_dim, emb_dim)
        self.ln_enc_emb = nn.LayerNorm(emb_dim)
        self.ln_dec_emb = nn.LayerNorm(emb_dim)
        self.encoder = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.decoder = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.ln_dec_out = nn.LayerNorm(hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src, trg_input):
        enc_emb = self.ln_enc_emb(self.embedding_enc(src))
        _, hidden = self.encoder(enc_emb)
        dec_emb = self.ln_dec_emb(self.embedding_dec(trg_input))
        dec_output, _ = self.decoder(dec_emb, hidden)
        dec_output = self.ln_dec_out(dec_output)
        return self.fc_out(dec_output)

# Training Utilities
def get_dir(base_dir="runs", prefix="train"):
    os.makedirs(base_dir, exist_ok=True)
    run_id = 0
    while os.path.exists(os.path.join(base_dir, f"{prefix}{run_id}")):
        run_id += 1
    run_dir = os.path.join(base_dir, f"{prefix}{run_id}")
    os.makedirs(run_dir)
    return run_dir

def train_epoch(model, dataloader, optimizer, criterion, device, log_file, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    start_time = time.time()  # Record start time for epoch

    for src, trg_input, trg_output in pbar:
        src, trg_input, trg_output = src.to(device), trg_input.to(device), trg_output.to(device)
        optimizer.zero_grad()
        output = model(src, trg_input)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg_output = trg_output[:, 1:].reshape(-1)
        loss = criterion(output, trg_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    epoch_time = time.time() - start_time  # Calculate epoch duration
    avg_loss = total_loss / len(dataloader)
    
    # Log loss and time to file
    with open(log_file, "a") as f:
        f.write(f"Epoch: {epoch}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s\n")
    
    return avg_loss

# Main Training Loop
def main():
    # Record start time of training
    training_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Load data
    zh_vocab, en_vocab = load_vocabs(SRC_PATH + "zh_vocab.pkl", SRC_PATH + "en_vocab.pkl")
    INPUT_DIM = len(zh_vocab)
    OUTPUT_DIM = len(en_vocab)
    encoded = load_encoded_data(SRC_PATH + "encoded_data.pkl")

    # Initialize dataset and dataloader
    dataset = EncodedDataset(encoded)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = Seq2SeqGRU(
        INPUT_DIM,
        OUTPUT_DIM,
        EMB_DIM,
        HID_DIM,
        NUM_LAYERS
    ).to(DEVICE)

    # Use DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(DEVICE)

    # Load and set pretrained embeddings
    zh_emb = torch.load(SRC_PATH + "zh_emb.pt", weights_only=True).to(DEVICE)
    en_emb = torch.load(SRC_PATH + "en_emb.pt", weights_only=True).to(DEVICE)
    model.module.embedding_enc.weight.data.copy_(zh_emb)
    model.module.embedding_dec.weight.data.copy_(en_emb)
    model.module.embedding_enc.weight.requires_grad = False
    model.module.embedding_dec.weight.requires_grad = False

    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training setup
    run_dir = get_dir()
    best_path = os.path.join(run_dir, "best.pt")
    final_path = os.path.join(run_dir, "last.pt")
    log_file = os.path.join(run_dir, "log.txt")  # Log file path
    best_loss = float("inf")

    # Write training start time to log
    with open(log_file, "a") as f:
        f.write(f"Training started at: {training_start_time}\n")

    # Training loop
    for epoch in range(1, N_EPOCHS + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, DEVICE, log_file, epoch)
        print(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.module.state_dict(), best_path)

    # Save final model
    torch.save(model.module.state_dict(), final_path)
    print(f"✅ Model saved as {final_path}")

if __name__ == "__main__":
    main()