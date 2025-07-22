# dataset.py
import pandas as pd
import re
import jieba
from collections import Counter
import pickle

DATASET = "E:\\Document\\DeepLearning\\dataset\\OpenSubtitlesSub2.en-zh.tsv"
GLOVE_PATH = "embedding/glove.6B.200d.txt"
TENCENT_PATH = "embedding/light_Tencent_AILab_ChineseEmbedding.txt"
SRC_PATH = "ch2enMulty/src/frac18"

MIN_FREQ = 10
MAX_LEN = 50

# 清洗函数
def clean_en(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

def clean_zh(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    return " ".join(jieba.cut(text))

# 构建原始词表
def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for sent in sentences:
        counter.update(sent.split())
    vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab

# 从 GloVe 文件加载词表
def load_glove_vocab(path):
    glove_vocab = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.split()[0]
            glove_vocab.add(word)
    return glove_vocab

# 从 Tencent 文件加载词表
def load_tencent_vocab(path):
    tencent_vocab = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()[0]
            tencent_vocab.add(word)
    return tencent_vocab

# 与 embedding 词表做交集，重新构建 vocab（保持特殊 token）
def intersect_vocab(local_vocab, emb_vocab):
    new_vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    for word in local_vocab:
        if word in emb_vocab and word not in new_vocab:
            new_vocab[word] = len(new_vocab)
    return new_vocab

# 编码句子
def encode_sentence(sentence, vocab, max_len):
    ids = [vocab.get(w, vocab["<unk>"]) for w in sentence.split()]
    ids = ids[:max_len - 2]
    ids = [vocab["<sos>"]] + ids + [vocab["<eos>"]]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids

def main():
    # 读取数据
    data = pd.read_csv(
        DATASET,
        sep="\t",
        names=["en", "zh"],
        on_bad_lines='skip',
        encoding="utf-8"
    )
    data.dropna(inplace=True)

    # 清洗
    data["en"] = data["en"].apply(clean_en)
    data["zh"] = data["zh"].apply(clean_zh)

    # 本地词表（含频率筛选）
    zh_vocab_raw = build_vocab(data["zh"], min_freq=MIN_FREQ)
    en_vocab_raw = build_vocab(data["en"], min_freq=MIN_FREQ)

    # 加载词向量的词表
    glove_vocab = load_glove_vocab(GLOVE_PATH)
    tencent_vocab = load_tencent_vocab(TENCENT_PATH)

    # 构建交集词表
    zh_vocab = intersect_vocab(zh_vocab_raw, tencent_vocab)
    en_vocab = intersect_vocab(en_vocab_raw, glove_vocab)

    print(f"中文词表（交集）大小：{len(zh_vocab)}")
    print(f"英文词表（交集）大小：{len(en_vocab)}")

    # 编码
    encoder_input = [encode_sentence(s, zh_vocab, MAX_LEN) for s in data["zh"]]
    decoder_input = [encode_sentence(s, en_vocab, MAX_LEN)[:-1] for s in data["en"]]
    decoder_target = [encode_sentence(s, en_vocab, MAX_LEN)[1:] for s in data["en"]]

    # 保存
    with open(SRC_PATH + "zh_vocab.pkl", "wb") as f:
        pickle.dump(zh_vocab, f)
    with open(SRC_PATH + "en_vocab.pkl", "wb") as f:
        pickle.dump(en_vocab, f)
    with open(SRC_PATH + "encoded_data.pkl", "wb") as f:
        pickle.dump({
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "decoder_target": decoder_target
        }, f)

    print("✅ 已保存词表和编码数据")

if __name__ == "__main__":
    main()
