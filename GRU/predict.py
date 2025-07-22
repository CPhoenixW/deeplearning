# predict.py
import torch
import torch.nn as nn
import pickle
import jieba
import re

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

# ---------------- 模型类 ---------------- #
class Seq2SeqGRU(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding_enc = nn.Embedding(input_dim, emb_dim)
        self.embedding_dec = nn.Embedding(output_dim, emb_dim)
        self.encoder = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.decoder = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src, trg_input):
        enc_emb = self.embedding_enc(src)
        _, hidden = self.encoder(enc_emb)

        dec_emb = self.embedding_dec(trg_input)
        output, _ = self.decoder(dec_emb, hidden)
        return self.fc_out(output)

# ---------------- 预测函数 ---------------- #
def translate_zh_to_en(text, model, zh_vocab, en_vocab, en_ivocab, max_len=40):
    model.eval()
    with torch.no_grad():
        input_tensor = encode_sentence(clean_zh(text), zh_vocab, max_len).to(model.fc_out.weight.device)
        encoder_emb = model.embedding_enc(input_tensor)
        _, hidden = model.encoder(encoder_emb)

        output_ids = [en_vocab["<sos>"]]
        for _ in range(max_len):
            prev_input = torch.tensor([[output_ids[-1]]], device=hidden.device)
            embedded = model.embedding_dec(prev_input)
            output, hidden = model.decoder(embedded, hidden)
            logits = model.fc_out(output[:, -1, :])
            next_id = logits.argmax(-1).item()
            if next_id == en_vocab["<eos>"]:
                break
            output_ids.append(next_id)

        translated = [en_ivocab.get(idx, "<unk>") for idx in output_ids[1:]]
        return " ".join(translated)

# ---------------- 主逻辑 ---------------- #
def main():
    MAX_LEN = 40
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取词表
    with open("../zh_vocab.pkl", "rb") as f:
        zh_vocab = pickle.load(f)
    with open("../en_vocab.pkl", "rb") as f:
        en_vocab = pickle.load(f)
    en_ivocab = {idx: word for word, idx in en_vocab.items()}

    # 模型结构
    model = Seq2SeqGRU(len(zh_vocab), len(en_vocab), 128, 256).to(DEVICE)
    model.load_state_dict(torch.load("gru_seq2seq.pth", map_location=DEVICE))

    # 测试翻译
    while True:
        zh = input("\n请输入中文句子（或输入q退出）：\n> ")
        if zh.lower() == "q":
            break
        en = translate_zh_to_en(zh, model, zh_vocab, en_vocab, en_ivocab, MAX_LEN)
        print("英文翻译结果：", en)

if __name__ == "__main__":
    main()
