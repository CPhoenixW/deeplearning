# train.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
from tqdm import tqdm


HID_DIM = 256
EMB_DIM = 200
BATCH_SIZE = 450
MAX_LEN = 50
N_EPOCHS = 150
NUM_LAYERS = 3
SRC_PATH = "src/frac14/"


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def get_dir(base_dir="runs", prefix="train"):
    os.makedirs(base_dir, exist_ok=True)
    run_id = 0
    while os.path.exists(os.path.join(base_dir, f"{prefix}{run_id}")):
        run_id += 1
    run_dir = os.path.join(base_dir, f"{prefix}{run_id}")
    os.makedirs(run_dir)
    return run_dir


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


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", disable=not is_main_process())

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
        if is_main_process():
            pbar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Load data
    zh_vocab, en_vocab = load_vocabs(SRC_PATH + "zh_vocab.pkl", SRC_PATH + "en_vocab.pkl")
    input_dim = len(zh_vocab)
    output_dim = len(en_vocab)
    encoded = load_encoded_data(SRC_PATH + "encoded_data.pkl")

    dataset = EncodedDataset(encoded)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    model = Seq2SeqGRU(input_dim, output_dim, EMB_DIM, HID_DIM, NUM_LAYERS).to(device)

    zh_emb = torch.load(SRC_PATH + "zh_emb.pt", map_location=device, weights_only=True)
    en_emb = torch.load(SRC_PATH + "en_emb.pt", map_location=device, weights_only=True)
    model.embedding_enc.weight.data.copy_(zh_emb)
    model.embedding_dec.weight.data.copy_(en_emb)
    model.embedding_enc.weight.requires_grad = False
    model.embedding_dec.weight.requires_grad = False

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    run_dir = get_dir() if is_main_process() else None
    best_path = os.path.join(run_dir, "best.pt") if run_dir else None
    final_path = os.path.join(run_dir, "last.pt") if run_dir else None
    best_loss = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        sampler.set_epoch(epoch)
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print0(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss and is_main_process():
            best_loss = avg_loss
            torch.save(model.module.state_dict(), best_path)

    if is_main_process():
        torch.save(model.module.state_dict(), final_path)
        print(f"✅ Model saved as {final_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
