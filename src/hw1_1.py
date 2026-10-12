# %% [markdown]
# # HW1 Problem 1 — N-Gram, RNN, and LSTM Models for Language Identification
#
# **任務**：對 English / French / Spanish / Portuguese / German 五種語言，各自訓練一個 character-level 語言模型，
# 對輸入文字計算在每個語言模型下的 perplexity，選擇 **perplexity 最低** 的語言作為預測。
#
# | 部分 | 內容 |
# |---|---|
# | 1 | Bigram / Trigram（Add-α, α = 1），只用 NumPy / pandas / collections / math / random 自行實作 |
# | 2 | 5 個 character-level RNN LM（PyTorch） |
# | 3 | 5 個 character-level LSTM LM（PyTorch，同 RNN 設定） |
# | 4 | 四種模型的綜合比較與分析 |
#
# **共同約定（所有模型一致）**
# - 字元詞彙表只由 `train.csv` 建立，所有語言共用；特殊 token：`<PAD>`=0, `<UNK>`=1, `<BOS>`=2, `<EOS>`=3。
# - 不在訓練詞彙中的字元 → `<UNK>`。
# - 每段文字 $x=(c_1,\dots,c_T)$ 轉成 `<BOS> c_1 … c_T <EOS>`；**預測目標為 $c_1,\dots,c_T,$ `<EOS>`**
#   （排除 `<BOS>` 與 `<PAD>`，包含 `<EOS>`），因此每段文字的有效目標數 $T_i = |x_i| + 1$。
# - Validation / Test 的 perplexity 一律使用 corpus-level 定義（每段文字用其**真實語言**的模型計算，先加總 NLL 與 token 數再取 exp），不平均個別文字的 PPL。
# - 使用 `train.csv` 訓練、`val.csv` 做模型 / checkpoint 選擇、`test.csv` 只做最終評估；不重抽樣、不重新切分資料。

# %% [markdown]
# ## 0. 環境設定與資料載入
#
# 將課程提供的 `train.csv`、`val.csv`、`test.csv` 放在 `DATA_DIR`（預設 `./data`）。若使用 Colab，可改成 Google Drive 路徑。

# %%
import os
import gc
import time
import math
import copy
import random
import tracemalloc
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
DATA_DIR = os.environ.get("HW1_DATA_DIR", "./data")
LANGS = ["English", "French", "Spanish", "Portuguese", "German"]
LANG2ID = {l: i for i, l in enumerate(LANGS)}
K = len(LANGS)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


set_seed()
pd.set_option("display.precision", 4)
pd.set_option("display.max_colwidth", 120)

# %%
def load_split(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    df["Text"] = df["Text"].fillna("").astype(str)
    df["Language"] = df["Language"].astype(str).str.strip()
    return df


train_df, val_df, test_df = load_split("train"), load_split("val"), load_split("test")
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    assert set(df["Language"]) == set(LANGS), f"{name}: unexpected labels {set(df['Language'])}"

summary = pd.DataFrame({
    name: df["Language"].value_counts().reindex(LANGS)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]
})
summary.loc["Total"] = summary.sum()
display(summary)

length_stats = pd.concat({
    name: df.assign(n_chars=df["Text"].str.len()).groupby("Language")["n_chars"].describe()[["mean", "50%", "min", "max"]]
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]
}, axis=1).reindex(LANGS)
print("每段文字的字元數統計")
display(length_stats)
display(train_df.groupby("Language").head(1)[["Language", "Text"]])

# %% [markdown]
# ### 共用字元詞彙表與 char ↔ index 轉換

# %%
PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
SPECIALS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(4)

char_freq = Counter(ch for text in train_df["Text"] for ch in text)
itos = SPECIALS + sorted(char_freq)            # index -> token
stoi = {tok: i for i, tok in enumerate(itos)}  # token -> index
V = len(itos)


def encode(text):
    return [stoi.get(ch, UNK_ID) for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


for df in (train_df, val_df, test_df):
    df["ids"] = df["Text"].map(encode)
    df["y"] = df["Language"].map(LANG2ID)
    df["n_chars"] = df["Text"].str.len()
    df["n_targets"] = df["n_chars"] + 1  # c_1..c_T + <EOS>

print(f"|V| = {V}  (含 {len(SPECIALS)} 個特殊 token，{V - len(SPECIALS)} 個字元)")
for name, df in [("val", val_df), ("test", test_df)]:
    n_unk = sum(ids.count(UNK_ID) for ids in df["ids"])
    print(f"{name}: OOV 字元 → <UNK> 的比例 = {n_unk / df['n_chars'].sum():.4%}")
sample = train_df["Text"].iloc[0][:40]
print("encode/decode 驗證:", repr(sample), "->", encode(sample)[:10], "...")
assert decode(encode(sample)) == sample

train_ids_by_lang = {l: train_df.loc[train_df["Language"] == l, "ids"].tolist() for l in LANGS}
val_ids_by_lang = {l: val_df.loc[val_df["Language"] == l, "ids"].tolist() for l in LANGS}

# %% [markdown]
# ### 評估工具（所有模型共用）
#
# 每個模型都輸出一個矩陣 `nll_sum[i, l]` = 第 i 段文字在語言 l 模型下的總 NLL，以及 `n_tok[i]` = $T_i$。
# - 預測：$\hat l_i = \arg\min_l \exp(\text{nll\_sum}[i,l]/T_i)$
# - Corpus-level PPL：$\exp\left(\sum_i \text{nll\_sum}[i, l_i] / \sum_i T_i\right)$，$l_i$ 為真實語言
# - Accuracy、Macro-F1、confusion matrix 皆以 NumPy 自行計算。

# %%
def confusion_matrix(y_true, y_pred, k=K):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm):
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1


def corpus_ppl(nll_sum, n_tok, y_true):
    """Corpus-level PPL: every text is scored by its ground-truth language model."""
    return math.exp(nll_sum[np.arange(len(y_true)), y_true].sum() / np.sum(n_tok))


def evaluate(name, nll_sum, n_tok, df):
    y_true = df["y"].to_numpy()
    text_ppl = np.exp(nll_sum / np.asarray(n_tok)[:, None])
    y_pred = text_ppl.argmin(1)
    cm = confusion_matrix(y_true, y_pred)
    p, r, f1 = per_class_prf(cm)
    res = {
        "model": name,
        "accuracy": np.trace(cm) / cm.sum(),
        "macro_f1": f1.mean(),
        "corpus_ppl": corpus_ppl(nll_sum, n_tok, y_true),
        "cm": cm, "y_true": y_true, "y_pred": y_pred, "text_ppl": text_ppl,
        "per_class": pd.DataFrame({"precision": p, "recall": r, "f1": f1}, index=LANGS),
    }
    print(f"[{name}] Accuracy = {res['accuracy']:.4f} | Macro-F1 = {res['macro_f1']:.4f} | "
          f"corpus-level PPL = {res['corpus_ppl']:.4f}")
    return res


def plot_cm(cm, title, ax=None):
    ax = ax or plt.subplots(figsize=(5.2, 4.4))[1]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(K), LANGS, rotation=45, ha="right")
    ax.set_yticks(range(K), LANGS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def confused_pairs(cm, top=5):
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            if cm[i, j] + cm[j, i] > 0:
                rows.append({"pair": f"{LANGS[i]} ↔ {LANGS[j]}", "i→j": cm[i, j], "j→i": cm[j, i], "total": cm[i, j] + cm[j, i]})
    if not rows:
        return pd.DataFrame(columns=["pair", "i→j", "j→i", "total"])
    return pd.DataFrame(rows).sort_values("total", ascending=False).head(top).reset_index(drop=True)


# 用 test 文字長度的五分位數切 bucket，所有模型共用同一組邊界
LEN_BINS = np.unique(np.quantile(test_df["n_chars"], np.linspace(0, 1, 6)))
LEN_BINS[0], LEN_BINS[-1] = -np.inf, np.inf


def length_report(res, df=test_df):
    d = pd.DataFrame({"n_chars": df["n_chars"].to_numpy(), "correct": res["y_true"] == res["y_pred"]})
    d["length bucket"] = pd.cut(d["n_chars"], LEN_BINS, precision=0)
    out = d.groupby("length bucket", observed=True)["correct"].agg(["count", "mean"])
    return out.rename(columns={"count": "n_texts", "mean": "accuracy"})


def show_examples(res, df=test_df, n=6, correct=False, seed=SEED):
    """顯示預測錯誤（或正確）的例子，並列出真實語言與預測語言的 PPL 差距。"""
    mask = (res["y_true"] == res["y_pred"]) == correct
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        print("（沒有符合的例子）")
        return None
    idx = np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False)
    ppl = res["text_ppl"]
    rows = []
    for i in idx:
        t, p = res["y_true"][i], res["y_pred"][i]
        row = {"len": df["n_chars"].iloc[i], "true": LANGS[t]}
        if correct:  # 和第二名的差距
            second = np.argsort(ppl[i])[1]
            row.update({"PPL(true)": ppl[i, t], "runner-up": LANGS[second], "PPL(runner-up)": ppl[i, second]})
        else:
            row.update({"pred": LANGS[p], "PPL(pred)": ppl[i, p], "PPL(true)": ppl[i, t],
                        "rank of true": int((ppl[i] < ppl[i, t]).sum()) + 1})
        row["text"] = df["Text"].iloc[i][:110]
        rows.append(row)
    return pd.DataFrame(rows)


def mb(n_bytes):
    return n_bytes / 2 ** 20

# %% [markdown]
# ---
# ## Part 1 — Character-level N-gram Language Models（Bigram / Trigram, Add-α, α = 1）
#
# $$p(c_t\mid c_{t-1},l)=\frac{C_l(c_{t-1},c_t)+\alpha}{C_l(c_{t-1})+\alpha|\mathcal V|},\qquad
#   p(c_t\mid c_{t-2},c_{t-1},l)=\frac{C_l(c_{t-2},c_{t-1},c_t)+\alpha}{C_l(c_{t-2},c_{t-1})+\alpha|\mathcal V|}$$
#
# **實作細節**
# - 序列前面補 $n-1$ 個 `<BOS>`、後面補 `<EOS>`：bigram 為 `<BOS> c_1 … c_T <EOS>`，trigram 為 `<BOS> <BOS> c_1 … c_T <EOS>`，
#   所以第一個字元的條件機率是 $p(c_1\mid\texttt{<BOS>})$ 或 $p(c_1\mid\texttt{<BOS>},\texttt{<BOS>})$。
# - $C_l(\text{context})$ 是該 context 作為「前文」出現的次數。因為每個字元後面一定接著一個目標（字元或 `<EOS>`），
#   這等於 context 的出現次數；這樣保證 $\sum_{c\in\mathcal V} p(c\mid\text{context})=1$。
# - $|\mathcal V|$ 為共用詞彙表大小（含特殊 token），與 RNN / LSTM 的 softmax 輸出維度一致。
# - 以 `defaultdict(int)` 計數；查詢時用 `.get()`，避免 defaultdict 在查詢時插入新 key 而讓記憶體變大。
# - 只用 `collections`、`math`、`numpy`、`pandas`，沒有用任何 n-gram 套件。

# %%
class NGramLM:
    """Character-level n-gram LM with add-alpha smoothing (implemented from scratch)."""

    def __init__(self, n, vocab_size, alpha=1.0):
        self.n, self.V, self.alpha = n, vocab_size, alpha
        self.ngram_counts = defaultdict(int)    # (c_{t-n+1}, ..., c_{t-1}, c_t) -> count
        self.context_counts = defaultdict(int)  # (c_{t-n+1}, ..., c_{t-1})      -> count

    def _pad(self, ids):
        return [BOS_ID] * (self.n - 1) + list(ids) + [EOS_ID]

    def fit(self, sequences):
        n = self.n
        for ids in sequences:
            s = self._pad(ids)
            for t in range(n - 1, len(s)):
                ctx = tuple(s[t - n + 1:t])
                self.ngram_counts[ctx + (s[t],)] += 1
                self.context_counts[ctx] += 1
        return self

    def prob(self, ctx, token):
        num = self.ngram_counts.get(ctx + (token,), 0) + self.alpha
        den = self.context_counts.get(ctx, 0) + self.alpha * self.V
        return num / den

    def nll(self, ids):
        """Return (total negative log-likelihood, number of prediction targets) of one text."""
        n, s = self.n, self._pad(ids)
        total = 0.0
        for t in range(n - 1, len(s)):
            total -= math.log(self.prob(tuple(s[t - n + 1:t]), s[t]))
        return total, len(s) - (n - 1)

    def num_entries(self):
        return len(self.ngram_counts) + len(self.context_counts)


# 驗證機率加總為 1（隨機挑一個看過的 context 與一個沒看過的 context）
_m = NGramLM(2, V).fit(train_ids_by_lang["English"])
for ctx in [(stoi.get("e", UNK_ID),), (UNK_ID,)]:
    assert abs(sum(_m.prob(ctx, c) for c in range(V)) - 1) < 1e-9
del _m
print("sum_c p(c | context) = 1 ✔")

# %%
ALPHA = 1.0


def build_ngram_models(n):
    return {l: NGramLM(n, V, ALPHA).fit(train_ids_by_lang[l]) for l in LANGS}


def ngram_score_matrix(models, ids_list):
    nll_sum = np.zeros((len(ids_list), K))
    n_tok = np.zeros(len(ids_list), dtype=np.int64)
    for i, ids in enumerate(ids_list):
        for j, l in enumerate(LANGS):
            nll_sum[i, j], n_tok[i] = models[l].nll(ids)
    return nll_sum, n_tok


def measure(fn):
    """Run fn twice: once for wall-clock time, once under tracemalloc for peak Python-heap memory."""
    gc.collect()
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    del out
    gc.collect()
    tracemalloc.start()
    out = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, elapsed, peak


ngram = {}
for n, name in [(2, "Bigram"), (3, "Trigram")]:
    models, build_t, build_peak = measure(lambda: build_ngram_models(n))
    ngram[name] = {"n": n, "models": models, "build_time": build_t, "build_peak": build_peak,
                   "entries": sum(m.num_entries() for m in models.values())}
    print(f"{name}: build time = {build_t:.2f}s, peak memory = {mb(build_peak):.1f} MB, "
          f"stored n-gram/context entries = {ngram[name]['entries']:,}")

# %% [markdown]
# ### 1(a) Validation corpus-level perplexity

# %%
val_rows = []
for name, d in ngram.items():
    nll_sum, n_tok = ngram_score_matrix(d["models"], val_df["ids"].tolist())
    d["val_nll"], d["val_ntok"] = nll_sum, n_tok
    y = val_df["y"].to_numpy()
    row = {"model": name, "val corpus PPL (all)": corpus_ppl(nll_sum, n_tok, y)}
    for j, l in enumerate(LANGS):
        m = y == j
        row[l] = math.exp(nll_sum[m, j].sum() / n_tok[m].sum())
    row["val accuracy"] = (np.exp(nll_sum / n_tok[:, None]).argmin(1) == y).mean()
    val_rows.append(row)
ngram_val_table = pd.DataFrame(val_rows).set_index("model")
print("Validation corpus-level PPL（各語言欄位 = 該語言 val 文字在自己語言模型下的 corpus PPL）")
display(ngram_val_table)

# %% [markdown]
# ### 1(b) Test：Accuracy、Macro-F1、corpus-level PPL、confusion matrix

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (name, d) in zip(axes, ngram.items()):
    ids_list = test_df["ids"].tolist()
    (nll_sum, n_tok), infer_t, infer_peak = measure(lambda: ngram_score_matrix(d["models"], ids_list))
    d["infer_time"], d["infer_peak"] = infer_t, infer_peak
    d["test"] = evaluate(name, nll_sum, n_tok, test_df)
    plot_cm(d["test"]["cm"], f"{name} — test", ax=ax)
plt.tight_layout()
plt.show()

for name, d in ngram.items():
    print(f"\n{name} per-class precision / recall / F1")
    display(d["test"]["per_class"])

# %% [markdown]
# ### 1(c) 建構時間、推論時間、記憶體使用與錯誤分析
#
# - 時間以 `time.perf_counter()` 量測（不含 tracemalloc 的額外負擔）。
# - 記憶體以 `tracemalloc` 另外跑一次量測 Python heap 的峰值（n-gram 完全在 CPU 上執行）。
# - 推論時間 = 對全部 test 文字在 5 個語言模型下都計算 PPL 的總時間。

# %%
ngram_cost = pd.DataFrame([{
    "model": name,
    "build time (s)": d["build_time"],
    "build peak mem (MB)": mb(d["build_peak"]),
    "stored entries": d["entries"],
    "test inference time (s)": d["infer_time"],
    "ms / text (5 models)": 1000 * d["infer_time"] / len(test_df),
    "inference peak mem (MB)": mb(d["infer_peak"]),
} for name, d in ngram.items()]).set_index("model")
display(ngram_cost)

for name, d in ngram.items():
    print(f"\n===== {name} =====")
    print("最常混淆的語言對（i→j = 真實為左邊語言卻預測成右邊語言）：")
    display(confused_pairs(d["test"]["cm"]))
    print("依文字長度（字元數）分組的 accuracy：")
    display(length_report(d["test"]))
    print("代表性的錯誤預測：")
    display(show_examples(d["test"]))

# %% [markdown]
# **1(c) 討論**
#
# > 以下數字來自一次完整執行的結果。在 GPU 上重跑後，accuracy / PPL 可能在小數點後略有不同，時間與記憶體一定會不同，請對照你的輸出更新。
#
# **建構 / 推論成本**
# - n-gram 的「訓練」只是把訓練資料掃一遍做計數：bigram 約 0.13 秒、trigram 約 0.18 秒。
# - 記憶體峰值 bigram 約 0.5 MB、trigram 約 2.9 MB。trigram 存的 entry 數（30,307）約是 bigram（5,855）的 5 倍，
#   因為 context 從單一字元變成字元對，可能的組合數大幅增加。
# - 推論只是查表：700 段 test 文字 × 5 個語言模型約 0.12 秒，每段約 0.18 ms，完全不需要 GPU。
#
# **哪些語言容易混淆**
# - **Spanish ↔ Portuguese 是最主要的錯誤**：bigram 19 個錯誤中有 12 個、trigram 9 個錯誤中有 5 個都是這一對。
# - 兩者同屬伊比利羅曼語，大量功能詞與字根拼法相同（`de`、`sobre`、`países`、`grupos`），所以 Portuguese 的 recall 最低（bigram 0.914）。
# - English 幾乎全對（F1 0.996）。一個原因是字元分布本來就和羅曼語差異大；另一個原因是本資料集的 English 文字平均長度約 160 字元，
#   是其他語言（約 40 字元）的 4 倍，證據比較多。
#
# **代表性錯誤**
# - `sobre países competitivos.`（Spanish → Portuguese）：每個字在兩種語言裡拼法都一樣，本質上無法只靠字元判斷。
# - `Afinal, Jimmy Stewart obstruiu durante`（Portuguese → German/English）：專有名詞 `Jimmy Stewart` 佔了大半字元，把判斷拉向日耳曼語。
# - `4.800 --`：幾乎只有數字與符號，沒有語言資訊。
# - `Ándale! Ándale!"` 標成 Portuguese，但 *ándale* 其實是西班牙文。這類**標註雜訊**，模型猜 Spanish 反而合理。
#
# **短文字是否比較難**
# - 是。bigram 在最短的長度區間（≤ 26 字元）accuracy 只有 92.3%，最長的區間（> 69 字元）是 100%。
# - 錯誤文字平均只有約 26 字元，正確文字約 69 字元。
# - 因為 English 都很長，Part 4 另外排除 English 再看一次，趨勢依然成立（92.8% → 100%），所以這是長度本身的效果，不是語言的效果。
#
# **Bigram vs Trigram**
# - trigram 整體較好：accuracy 97.29% → 98.71%，test PPL 11.33 → 9.40。
# - 但看各語言的 val PPL，只有 English（10.74 → 7.25）大幅下降，Spanish（11.99 → 13.63）、Portuguese（12.52 → 13.89）、German（12.61 → 13.61）反而**上升**。
# - 原因是 Add-1 平滑：每個 context 的分母都加上 |V| = 129。羅曼語與德語的訓練文字很短（每種語言約 4 萬字元），
#   大部分 trigram context 只出現幾次，加上 129 之後機率被攤得很平，PPL 變高。English 訓練資料多約 4 倍，trigram 才得以發揮。
# - 即使如此，trigram 的分類仍然比較準，這是「PPL 較低不一定代表 accuracy 較高」（反過來也是）的一個例子，Part 4 會再討論。

# %% [markdown]
# ---
# ## Part 2 & 3 — Character-level RNN / LSTM Language Models（PyTorch）
#
# 兩者共用同一套程式，只差 recurrent layer（`nn.RNN` with tanh vs. `nn.LSTM`）：
#
# `Embedding(|V|, 128, padding_idx=<PAD>) → {RNN | LSTM}(128, 128, num_layers=2) → Linear(128, |V|)`
#
# - 輸入 `<BOS> c_1 … c_T`，目標 `c_1 … c_T <EOS>`（next-character prediction），與 n-gram 的目標完全一致。
# - Loss：`CrossEntropyLoss(ignore_index=<PAD>)`，所以 `<PAD>` 不算在 loss / PPL 裡；`<BOS>` 從不是目標。
# - Optimizer：Adam, lr = 0.001；10 epochs；batch size 32；gradient clipping = 1.0（避免 RNN 梯度爆炸）。
# - 每個 epoch 結束後計算該語言 val 文字的 corpus-level PPL，**保留 val PPL 最低的 epoch 作為最終 checkpoint**。
# - **模型設定選擇**：除了作業建議的設定（lr = 0.001），另試 lr = 0.003（`HP_GRID`），以 5 個語言合併的 val corpus-level PPL 選出最終設定；
#   會試較大的 learning rate 是因為在 10 個 epoch 內 val PPL 仍持續下降（underfitting），此時調大 lr 比加 dropout 等 regularization 更有幫助；
#   test set 完全不參與選擇。
# - 依作業規定 RNN / LSTM 必須在 GPU 上執行。

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "⚠ 沒有偵測到 GPU（作業要求使用 GPU）")
print("torch", torch.__version__)

BASE_HP = dict(emb_dim=128, hidden_dim=128, num_layers=2, lr=1e-3, epochs=10, batch_size=32,
               dropout=0.0, clip=1.0)
# 以 val 選擇的候選設定：第一個為作業建議設定
HP_GRID = [dict(BASE_HP), dict(BASE_HP, lr=3e-3)]
EVAL_BATCH = 128

# %%
class CharLM(nn.Module):
    def __init__(self, vocab_size, cell, emb_dim, hidden_dim, num_layers, dropout=0.0, **_):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):                      # x: (B, L) token ids
        h, _ = self.rnn(self.dropout(self.embedding(x)))
        return self.fc(self.dropout(h))        # (B, L, |V|) next-token logits


def to_pair(ids):
    seq = [BOS_ID] + list(ids) + [EOS_ID]
    return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


def collate(batch):
    xs, ys = zip(*batch)
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=PAD_ID)
    y = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=PAD_ID)
    return x, y


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def nll_per_text(model, ids_list, batch_size=EVAL_BATCH):
    """Total NLL and number of targets for each text (texts are batched by length for speed)."""
    model.eval()
    nll = np.zeros(len(ids_list))
    ntok = np.zeros(len(ids_list), dtype=np.int64)
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        x, y = collate([to_pair(ids_list[i]) for i in idx])
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x).transpose(1, 2), y, ignore_index=PAD_ID, reduction="none")
        nll[idx] = loss.double().sum(1).cpu().numpy()
        ntok[idx] = (y != PAD_ID).sum(1).cpu().numpy()
    return nll, ntok


def train_lang_model(cell, lang, hp, seed=SEED):
    set_seed(seed)
    model = CharLM(V, cell, **hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loader = DataLoader([to_pair(ids) for ids in train_ids_by_lang[lang]], batch_size=hp["batch_size"],
                        shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    hist = {"step_loss": [], "epoch_loss": [], "val_ppl": []}
    best_ppl, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        tot_loss, tot_tok = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x).transpose(1, 2), y, ignore_index=PAD_ID)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["clip"])
            opt.step()
            n = (y != PAD_ID).sum().item()
            tot_loss += loss.item() * n
            tot_tok += n
            hist["step_loss"].append(loss.item())
        hist["epoch_loss"].append(tot_loss / tot_tok)
        v_nll, v_tok = nll_per_text(model, val_ids_by_lang[lang])
        hist["val_ppl"].append(math.exp(v_nll.sum() / v_tok.sum()))
        if hist["val_ppl"][-1] < best_ppl:
            best_ppl, best_epoch = hist["val_ppl"][-1], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    hist["best_epoch"], hist["best_val_ppl"] = best_epoch, best_ppl
    return model, hist


def train_family(cell, hp):
    """Train the 5 language models of one family with one hyper-parameter setting."""
    models, hists, train_time, train_peak = {}, {}, {}, {}
    for lang in LANGS:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        sync()
        t0 = time.perf_counter()
        models[lang], hists[lang] = train_lang_model(cell, lang, hp)
        sync()
        train_time[lang] = time.perf_counter() - t0
        train_peak[lang] = torch.cuda.max_memory_allocated() if device.type == "cuda" else float("nan")
        print(f"  [{cell.upper()} {lang:<10}] {train_time[lang]:6.1f}s | best epoch {hists[lang]['best_epoch']:2d} "
              f"| best val PPL {hists[lang]['best_val_ppl']:.4f}")
    # 5 個語言合併的 val corpus-level PPL（每個語言用自己的最佳 checkpoint）
    tot_nll = tot_tok = 0
    for lang in LANGS:
        v_nll, v_tok = nll_per_text(models[lang], val_ids_by_lang[lang])
        tot_nll, tot_tok = tot_nll + v_nll.sum(), tot_tok + v_tok.sum()
    return {"cell": cell, "hp": hp, "models": models, "hists": hists, "train_time": train_time,
            "train_peak": train_peak, "val_ppl": math.exp(tot_nll / tot_tok),
            "n_params": sum(p.numel() for p in models[LANGS[0]].parameters())}


def score_matrix_nn(models, ids_list):
    nll_sum = np.zeros((len(ids_list), K))
    n_tok = None
    for j, lang in enumerate(LANGS):
        nll_sum[:, j], n_tok = nll_per_text(models[lang], ids_list)
    return nll_sum, n_tok


def run_family(cell):
    runs = []
    for hp in HP_GRID:
        print(f"== {cell.upper()} | lr={hp['lr']} ==")
        runs.append(train_family(cell, hp))
    sel = min(runs, key=lambda r: r["val_ppl"])
    display(pd.DataFrame([{"lr": r["hp"]["lr"], "val corpus PPL (5 langs)": r["val_ppl"],
                           "total train time (s)": sum(r["train_time"].values()),
                           "selected": r is sel} for r in runs]))
    for r in runs:  # 釋放沒被選上的模型
        if r is not sel:
            r.pop("models")
    sel["grid"] = runs
    return sel


def plot_curves(res, title):
    hp = res["hp"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for lang in LANGS:
        h = res["hists"][lang]
        steps = np.asarray(h["step_loss"])
        win = max(1, len(steps) // (hp["epochs"] * 4))
        smooth = np.convolve(steps, np.ones(win) / win, mode="valid")
        axes[0].plot(np.arange(len(smooth)) / (len(steps) / hp["epochs"]), smooth, label=lang)
        ep = np.arange(1, hp["epochs"] + 1)
        axes[1].plot(ep, h["epoch_loss"], marker="o", label=lang)
        axes[2].plot(ep, h["val_ppl"], marker="o", label=lang)
        axes[2].scatter([h["best_epoch"]], [h["best_val_ppl"]], s=120, facecolors="none", edgecolors="k")
    axes[0].set(title="Training loss (per step, smoothed)", xlabel="epoch", ylabel="cross-entropy")
    axes[1].set(title="Training loss (per epoch)", xlabel="epoch", ylabel="cross-entropy")
    axes[2].set(title="Validation corpus-level PPL (○ = selected)", xlabel="epoch", ylabel="PPL")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{title} (lr = {hp['lr']})")
    plt.tight_layout()
    plt.show()


def test_family(res, name):
    ids_list = test_df["ids"].tolist()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.perf_counter()
    nll_sum, n_tok = score_matrix_nn(res["models"], ids_list)
    sync()
    res["infer_time"] = time.perf_counter() - t0
    res["infer_peak"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else float("nan")
    res["test"] = evaluate(name, nll_sum, n_tok, test_df)
    return res


def cost_table(res, name):
    return pd.DataFrame([{
        "model": name,
        "params / language model": res["n_params"],
        "total train time (s)": sum(res["train_time"].values()),
        "train peak GPU mem (MB)": mb(max(res["train_peak"].values())),
        "test inference time (s)": res["infer_time"],
        "ms / text (5 models)": 1000 * res["infer_time"] / len(test_df),
        "inference peak GPU mem (MB)": mb(res["infer_peak"]),
    }]).set_index("model")

# %% [markdown]
# ### Part 2 — RNN
# #### 2(a) 訓練與 learning curves（training loss）/ validation perplexity curves

# %%
rnn = run_family("rnn")
plot_curves(rnn, "RNN")
display(pd.DataFrame({l: {"best epoch": rnn["hists"][l]["best_epoch"],
                          "best val PPL": rnn["hists"][l]["best_val_ppl"],
                          "train time (s)": rnn["train_time"][l]} for l in LANGS}).T)

# %% [markdown]
# #### 2(b) 以 val 選出的設定與 checkpoint 在 test 上評估

# %%
rnn = test_family(rnn, "RNN")
display(rnn["test"]["per_class"])

# %% [markdown]
# #### 2(c) Confusion matrix、訓練 / 推論時間、記憶體、正確與錯誤預測

# %%
plot_cm(rnn["test"]["cm"], "RNN — test")
plt.show()
display(cost_table(rnn, "RNN"))
print("最常混淆的語言對：")
display(confused_pairs(rnn["test"]["cm"]))
print("依文字長度分組的 accuracy：")
display(length_report(rnn["test"]))
print("代表性的正確預測：")
display(show_examples(rnn["test"], correct=True))
print("代表性的錯誤預測：")
display(show_examples(rnn["test"]))

# %% [markdown]
# **2 討論**
#
# > 以下數字來自一次完整執行的結果。在 GPU 上重跑後，accuracy / PPL 可能在小數點後略有不同，時間與記憶體一定會不同，請對照你的輸出更新。
#
# **Learning curves 與模型選擇**
# - 5 個語言的 training loss 都持續下降，val PPL 也一路下降，最佳 checkpoint 全部落在第 10 個 epoch。
#   代表模型在 10 個 epoch 內還沒有 overfit，反而是**還沒訓練完全（underfitting）**。
# - 因此在作業建議的 lr = 0.001 之外，另試了 lr = 0.003：5 個語言合併的 val PPL 從 5.83 降到 4.94，所以選用 lr = 0.003。
#   在 underfitting 的情況下加 dropout 只會更糟（之前試過 dropout = 0.2，val PPL 反而上升）。
#
# **Test 結果**
# - Accuracy 98.86%、Macro-F1 98.85%、corpus-level PPL 4.76，三項都是 4 種模型中最好的。
# - 700 段只錯 8 段，其中 5 段是 Spanish ↔ Portuguese。
#
# **時間與記憶體**
# - 每個語言模型約 9.9 萬個參數。
# - English 的訓練時間是其他語言的 6～7 倍，因為 English 文字很長（最長 1,334 字元），每個 batch 要補齊到最長那段，序列運算步數多。
# - 推論需要對每段文字跑 5 次前向傳播，比 n-gram 查表慢約 8 倍。實際的 GPU 時間與記憶體請看上方表格。
#
# **代表性的正確預測**
# - 長的 English 文字：PPL(English) ≈ 3.0，第二名 French ≈ 36，差距非常大。
# - `existen estas señales de aquí.`：Spanish PPL 5.8，Portuguese 10.6。`ñ` 這類只出現在西班牙文的字元提供了關鍵證據。
#
# **代表性的錯誤預測**
# - `de prevenir o desastre climático,`（Portuguese → Spanish）：兩者 PPL 只差 7.89 vs 8.26。除了冠詞 `o` 之外，每個字西葡兩語都通用。
# - `grupos de gente feliz e infeliz` 標成 Spanish，但連接詞 `e` 是葡萄牙文用法（西文是 `y`），模型預測 Portuguese 其實有道理，可能是標註雜訊。
# - `4.800 --`、`Ándale! Ándale!"`：與 n-gram 相同的困難例子（沒有語言資訊、標註可疑）。

# %% [markdown]
# ### Part 3 — LSTM
# #### 3(a) 訓練與 learning curves / validation perplexity curves

# %%
lstm = run_family("lstm")
plot_curves(lstm, "LSTM")
display(pd.DataFrame({l: {"best epoch": lstm["hists"][l]["best_epoch"],
                          "best val PPL": lstm["hists"][l]["best_val_ppl"],
                          "train time (s)": lstm["train_time"][l]} for l in LANGS}).T)

# %% [markdown]
# #### 3(b) 以 val 選出的設定與 checkpoint 在 test 上評估

# %%
lstm = test_family(lstm, "LSTM")
display(lstm["test"]["per_class"])

# %% [markdown]
# #### 3(c) Confusion matrix、訓練 / 推論時間、記憶體、正確與錯誤預測

# %%
plot_cm(lstm["test"]["cm"], "LSTM — test")
plt.show()
display(cost_table(lstm, "LSTM"))
print("最常混淆的語言對：")
display(confused_pairs(lstm["test"]["cm"]))
print("依文字長度分組的 accuracy：")
display(length_report(lstm["test"]))
print("代表性的正確預測：")
display(show_examples(lstm["test"], correct=True))
print("代表性的錯誤預測：")
display(show_examples(lstm["test"]))

# %% [markdown]
# **3 討論**
#
# > 以下數字來自一次完整執行的結果。在 GPU 上重跑後，accuracy / PPL 可能在小數點後略有不同，時間與記憶體一定會不同，請對照你的輸出更新。
#
# **Learning curves 與模型選擇**
# - 情況和 RNN 一樣：val PPL 到第 10 個 epoch 都還在下降，最佳 checkpoint 都是第 10 個 epoch。
# - lr = 0.003 讓合併 val PPL 從 8.44 降到 6.47，因此選用 lr = 0.003。
#
# **Test 結果**
# - Accuracy 98.57%、Macro-F1 98.57%、corpus-level PPL 6.32。
#
# **為什麼 LSTM 沒有比 RNN 好？**
# 理論上 LSTM 記憶力較好，但這次 PPL（6.32 vs 4.76）與 accuracy（98.57% vs 98.86%）都輸給 RNN，原因有兩個：
# 1. **訓練預算不足**：LSTM 每層有 4 組 gate，參數量約 29.7 萬，是 RNN 的 3 倍，在相同的 10 個 epoch（每個語言約 350 次更新）內收斂得比較慢。
#    從曲線看 val PPL 仍在快速下降，增加 epoch 數應能超越 RNN。
# 2. **文字太短**：非 English 文字中位數只有約 40 字元，LSTM 擅長的長距離記憶派不上用場；字元層級的語言特徵大多在前後幾個字元內就能看到。
#
# **時間與記憶體**
# - 參數量是 RNN 的 3 倍，推論時間約為 RNN 的 1.5 倍；GPU 記憶體也較高（見上方表格）。
#
# **代表性的正確與錯誤預測**
# - 正確：`Er führt durch Städte.` German PPL 9.1，第二名 French 高達 108。`ü`、`ä` 與 `Er`/`durch` 等字讓判斷非常明確。
# - 錯誤：10 個錯誤中有 7 個是 Spanish ↔ Portuguese，例如 `tratando de ser valiente`（Spanish → Portuguese，PPL 6.22 vs 6.48）
#   和 `A cobertura jornalística`（Portuguese → Spanish）。這些都是短句，而且單字在兩種語言中拼法接近。

# %% [markdown]
# ---
# ## Part 4 — Bigram / Trigram / RNN / LSTM 綜合比較

# %%
all_res = {"Bigram": ngram["Bigram"]["test"], "Trigram": ngram["Trigram"]["test"],
           "RNN": rnn["test"], "LSTM": lstm["test"]}
val_ppl = {"Bigram": ngram_val_table.loc["Bigram", "val corpus PPL (all)"],
           "Trigram": ngram_val_table.loc["Trigram", "val corpus PPL (all)"],
           "RNN": rnn["val_ppl"], "LSTM": lstm["val_ppl"]}
gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

rows = []
for name in ["Bigram", "Trigram"]:
    d = ngram[name]
    rows.append({"model": name, "hardware": "CPU (pure Python)", "size": f"{d['entries']:,} count entries",
                 "train/build time (s)": d["build_time"], "train peak mem (MB)": mb(d["build_peak"]),
                 "test inference time (s)": d["infer_time"], "inference peak mem (MB)": mb(d["infer_peak"])})
for name, r in [("RNN", rnn), ("LSTM", lstm)]:
    rows.append({"model": name, "hardware": f"{device.type.upper()} ({gpu})", "size": f"{5 * r['n_params']:,} params (5 models)",
                 "train/build time (s)": sum(r["train_time"].values()),
                 "train peak mem (MB)": mb(max(r["train_peak"].values())),
                 "test inference time (s)": r["infer_time"], "inference peak mem (MB)": mb(r["infer_peak"])})
comparison = pd.DataFrame(rows).set_index("model")
for name, res in all_res.items():
    comparison.loc[name, "val corpus PPL"] = val_ppl[name]
    comparison.loc[name, "test corpus PPL"] = res["corpus_ppl"]
    comparison.loc[name, "test accuracy"] = res["accuracy"]
    comparison.loc[name, "test macro-F1"] = res["macro_f1"]
print("記憶體：n-gram 為 tracemalloc 量到的 Python heap 峰值；RNN/LSTM 為 torch.cuda.max_memory_allocated 的 GPU 峰值。")
display(comparison)

# %%
fig, axes = plt.subplots(1, 4, figsize=(22, 4.4))
for ax, (name, res) in zip(axes, all_res.items()):
    plot_cm(res["cm"], f"{name} — test", ax=ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 較低的 perplexity 是否一定代表較高的辨識 accuracy？
#
# 下表比較模型之間（以及同一模型內各語言之間）PPL 與 accuracy / F1 的排序。

# %%
rank = comparison[["test corpus PPL", "test accuracy", "test macro-F1"]].copy()
rank["PPL rank (1 = lowest)"] = rank["test corpus PPL"].rank().astype(int)
rank["accuracy rank (1 = highest)"] = rank["test accuracy"].rank(ascending=False, method="min").astype(int)
display(rank)

own_ppl, own_f1 = {}, {}
for name, res in all_res.items():
    y = res["y_true"]
    own_ppl[name] = [math.exp(np.log(res["text_ppl"][y == j, j]).mean()) for j in range(K)]
    own_f1[name] = res["per_class"]["f1"].to_numpy()
print("各語言：test 文字在自己語言模型下的 PPL（逐文字 PPL 的幾何平均，僅供比較排序）")
display(pd.DataFrame(own_ppl, index=LANGS))
print("各語言 test F1")
display(pd.DataFrame(own_f1, index=LANGS))

# %% [markdown]
# ### 哪些語言對最常被混淆？

# %%
pair_tbl = {}
for name, res in all_res.items():
    cm = res["cm"]
    for i in range(K):
        for j in range(i + 1, K):
            pair_tbl.setdefault(f"{LANGS[i]} ↔ {LANGS[j]}", {})[name] = cm[i, j] + cm[j, i]
pair_tbl = pd.DataFrame(pair_tbl).T
pair_tbl["total"] = pair_tbl.sum(axis=1)
display(pair_tbl.sort_values("total", ascending=False))

# %% [markdown]
# ### 文字較短時表現是否下降？

# %%
len_tbl = pd.concat({name: length_report(res)["accuracy"] for name, res in all_res.items()}, axis=1)
len_tbl.insert(0, "n_texts", length_report(all_res["Bigram"])["n_texts"])
display(len_tbl)

# 注意：各語言的文字長度分布不同（English 明顯較長），所以也要看每個長度 bucket 裡的語言組成，避免把「語言」的效果誤認為「長度」的效果
bucket = pd.cut(test_df["n_chars"], LEN_BINS, precision=0)
print("各長度 bucket 的語言組成（test）")
display(pd.crosstab(bucket, test_df["Language"]).reindex(columns=LANGS))

# 排除 English 後再看一次長度效應（只比較長度分布相近的 4 個語言）
non_en = (test_df["Language"] != "English").to_numpy()
len_tbl_no_en = pd.concat({
    name: pd.DataFrame({"b": bucket[non_en], "c": (res["y_true"] == res["y_pred"])[non_en]})
            .groupby("b", observed=True)["c"].mean()
    for name, res in all_res.items()}, axis=1)
print("排除 English 後，各長度 bucket 的 accuracy")
display(len_tbl_no_en)

ax = len_tbl.drop(columns="n_texts").plot(marker="o", figsize=(8, 4))
ax.set(title="Test accuracy vs. text length (character-count quintiles)", xlabel="text length (#chars)", ylabel="accuracy")
ax.grid(alpha=0.3)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

for name, res in all_res.items():
    wrong = res["y_true"] != res["y_pred"]
    if wrong.any():
        print(f"{name:8s}: 錯誤文字平均長度 = {test_df['n_chars'][wrong].mean():7.1f} 字元 | "
              f"正確文字平均長度 = {test_df['n_chars'][~wrong].mean():7.1f} 字元")
    else:
        print(f"{name:8s}: 沒有錯誤")

# %% [markdown]
# **Part 4 綜合討論**
#
# > 以下數字來自一次完整執行的結果。在 GPU 上重跑後，accuracy / PPL 可能在小數點後略有不同，時間與記憶體一定會不同，請對照你的輸出更新。
#
# | | Bigram | Trigram | RNN | LSTM |
# |---|---|---|---|---|
# | Test accuracy | 97.29% | 98.71% | **98.86%** | 98.57% |
# | Test macro-F1 | 97.27% | 98.71% | **98.85%** | 98.57% |
# | Test corpus PPL | 11.33 | 9.40 | **4.76** | 6.32 |
# | 錯誤數（/700） | 19 | 9 | **8** | 10 |
#
# **1. 硬體資源**
# - n-gram 只需要 CPU 和幾 MB 的記憶體，模型就是一張計數表（bigram 5,855 項、trigram 30,307 項）。
# - RNN / LSTM 需要 GPU 做反向傳播。5 個模型合計參數 RNN 約 50 萬、LSTM 約 149 萬，GPU 記憶體也以 LSTM 最高。
#
# **2. 計算時間**
# - n-gram 建構不到 1 秒；RNN / LSTM 每個設定要訓練 10 個 epoch，加上 val 選擇時跑了 2 組 lr，時間多了好幾個數量級。
# - 推論方面，n-gram 每段約 0.2 ms，RNN 與 LSTM 慢了數倍到十倍左右。
#
# **3. 效能**
# - RNN 在 PPL、accuracy、macro-F1 三項都最好。
# - 但所有模型的 accuracy 都在 97% 以上，差距只有幾段文字，**投入的運算成本與換來的 accuracy 不成比例**：trigram 只用不到 1 秒，accuracy 就只比 RNN 少 0.15%。
#
# **4. PPL 較低是否一定 accuracy 較高？不一定。**
# - **反例一**：LSTM 的 PPL（6.32）明顯低於 trigram（9.40），accuracy 卻較低（98.57% vs 98.71%）。
# - **反例二**：trigram 在 Spanish / Portuguese / German 的 val PPL 都比 bigram 高，這三種語言的 F1 卻都比 bigram 好。
# - **原因**：corpus PPL 衡量的是「真實語言的模型」預測自己語言文字的能力，是**絕對**的品質；
#   分類看的卻是真實語言的 PPL 是否**相對**低於其他 4 個模型。PPL 整體下降時，如果相近語言（如 Spanish 與 Portuguese）的模型同時變好，
#   兩者的差距不一定拉大，分類邊界就不會更清楚。
# - 此外 accuracy 已接近飽和，剩下的錯誤多是本質上模稜兩可或標註有問題的文字，PPL 再低也救不回來。
# - 整體趨勢上 PPL 較低的模型通常比較準（Bigram 最差、RNN 最好），但兩者不是嚴格單調的關係。
#
# **5. 最常混淆的語言對**
# - **Spanish ↔ Portuguese** 在 4 個模型的 46 個錯誤中佔了 29 個（63%），每個模型都是第一名。原因是兩者親緣最近，共用大量字根與功能詞。
# - 第二名 Portuguese ↔ German（7 個）並不是語言相似，而是含專有名詞、數字的文字（如 `Jimmy Stewart`、`4.800 --`），字元統計被非語言內容主導。
# - English ↔ German 雖同屬日耳曼語，但書寫的字元分布差異大，0 次混淆。
#
# **6. 文字較短時效能是否下降？是。**
# - 所有模型都在最短的長度區間（≤ 26 字元）表現最差：bigram 92.3%、LSTM 95.1%、RNN 97.2%。
#   超過 47 字元後，trigram / RNN / LSTM 都是 100%。
# - 錯誤文字平均長度約 24～26 字元，正確文字約 68 字元。
# - 因為 English 文字普遍很長、最長的區間有 101/140 段是 English，所以另外排除 English 再看，趨勢不變。這確實是長度造成的，不是語言造成的。
# - 神經網路對短文字的改善最多（≤ 26 字元：bigram 92.3% → RNN 97.2%），因為它能把整段字元的上下文一起納入判斷。
#
# **7. 資料集觀察**
# - 本資料集的 English 文字明顯比其他語言長，且內容多為宗教經文翻譯，和其他語言的 TED 演講逐字稿來源不同。
#   所以 English 在所有模型中都幾乎全對，一部分可能是「文字長度 / 文體」本身就成了線索，而不只是語言特徵。
