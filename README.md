# LLM HW1

| 檔案 | 內容 |
|---|---|
| `hw1_1_StudentID.ipynb` | Problem 1：character-level Bigram / Trigram / RNN / LSTM 語言辨識（繳交前把 `StudentID` 改成學號） |
| `src/hw1_1.py` | 同一份 notebook 的 jupytext 原始碼（方便看 diff）；`jupytext --to ipynb src/hw1_1.py -o hw1_1_StudentID.ipynb` 可重新產生 |
| `results/hw1_1_example_run.ipynb` | 用課程資料完整跑過一次的版本（含所有表格與圖；在 CPU 上跑的，時間 / GPU 記憶體數字僅供參考） |
| `data/` | 放課程資料 `train.csv` / `val.csv` / `test.csv`（不進版控） |

## 執行
RNN / LSTM 依規定需在 GPU 上執行（例如 Colab → Runtime → GPU）。需要 `numpy pandas matplotlib torch`。
