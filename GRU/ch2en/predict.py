# predict.py
import torch
import torch.nn as nn
import pickle
import jieba
import re

MODEL = r"E:\Document\DeepLearning\work\GRU\ch2en\runs\train2\best.pt"
SRC_PATH = "src/frac14/"


# Data Handling
def load_vocabs(zh_path, en_path):
    with open(zh_path, "rb") as f:
        zh_vocab = pickle.load(f)
    with open(en_path, "rb") as f:
        en_vocab = pickle.load(f)
    return zh_vocab, en_vocab

# ---------------- 清洗函数 ---------------- #
def clean_zh(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    return " ".join(jieba.cut(text))

# ---------------- 编码函数 ---------------- #
def encode_sentence(sentence, vocab, max_len):
    ids = [vocab.get(w, vocab["<unk>"]) for w in sentence.split()]
    ids = ids[:max_len - 2]
    ids = [vocab["<sos>"]] + ids + [vocab["<eos>"]]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor([ids])

# ---------------- 模型类（新版） ---------------- #
class Seq2SeqGRU(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, num_layers=3):
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
        output, _ = self.decoder(dec_emb, hidden)

        output = self.ln_dec_out(output)
        return self.fc_out(output)

# ---------------- 预测函数 ---------------- #
def translate_zh_to_en(text, model, zh_vocab, en_vocab, en_ivocab, max_len=40):
    model.eval()
    with torch.no_grad():
        input_tensor = encode_sentence(clean_zh(text), zh_vocab, max_len).to(next(model.parameters()).device)
        enc_emb = model.ln_enc_emb(model.embedding_enc(input_tensor))
        _, hidden = model.encoder(enc_emb)

        output_ids = [en_vocab["<sos>"]]
        for _ in range(max_len):
            prev_input = torch.tensor([[output_ids[-1]]], device=hidden.device)
            embedded = model.ln_dec_emb(model.embedding_dec(prev_input))
            output, hidden = model.decoder(embedded, hidden)
            output = model.ln_dec_out(output)
            logits = model.fc_out(output[:, -1, :])
            next_id = logits.argmax(-1).item()
            if next_id == en_vocab["<eos>"]:
                break
            output_ids.append(next_id)

        translated = [en_ivocab.get(idx, "<unk>") for idx in output_ids[1:]]
        return " ".join(translated)

# ---------------- 主逻辑 ---------------- #
def main():
    MAX_LEN = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取词表
    zh_vocab, en_vocab = load_vocabs(SRC_PATH + "zh_vocab.pkl", SRC_PATH + "en_vocab.pkl")
    en_ivocab = {idx: word for word, idx in en_vocab.items()}

    # 加载模型
    EMB_DIM = 200
    model = Seq2SeqGRU(len(zh_vocab), len(en_vocab), EMB_DIM, 256, num_layers=3).to(DEVICE)

    # 加载 embedding 向量
    zh_emb = torch.load(SRC_PATH + "zh_emb.pt", weights_only=True).to(DEVICE)
    en_emb = torch.load(SRC_PATH + "en_emb.pt", weights_only=True).to(DEVICE)
    model.embedding_enc.weight.data.copy_(zh_emb)
    model.embedding_dec.weight.data.copy_(en_emb)
    model.load_state_dict(torch.load(MODEL, map_location=DEVICE, weights_only=True))
    print("✅ 已加载最优模型 " + MODEL)

    # 测试翻译
    while True:
        zh = input("中文: \n> ")
        if zh.lower() == "q":
            break
        en = translate_zh_to_en(zh, model, zh_vocab, en_vocab, en_ivocab, MAX_LEN)
        print("Translation: ", en)

if __name__ == "__main__":
    main()
