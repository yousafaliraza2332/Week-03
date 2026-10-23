# Week 3 – Supervised Models on Urdu News (FAKE vs TRUE)

**Dataset:** Urdu news dataset, file `urdu_news.csv` (10,106 raw rows → 9,235 after cleaning; columns: `Sr. No.`, `News Items`, `Label`).
Source: _[add the original source / link here]_

## Summary
- The goal is to predict if an Urdu news item is **FAKE** or **TRUE**. The text is turned into 502 numbers (500 TF-IDF scores + word count + character count).
- I trained KNN, Decision Tree, Logistic Regression and Linear Regression (cut at 0.5) and compared them with accuracy, confusion matrix, precision, recall and F1.
- Classes are balanced (50.2 % FAKE / 49.8 % TRUE). Best K = 5, best `max_depth` = 10; very deep trees overfit (100 % train vs 77 % test).

## Best model
**Logistic Regression** – accuracy **83.2 %**, F1 (FAKE) **0.830**.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.832 | 0.842 | 0.818 | 0.830 |
| Linear Regression (≥ 0.5) | 0.808 | 0.830 | 0.776 | 0.802 |
| Decision Tree (depth 10) | 0.779 | 0.805 | 0.739 | 0.771 |
| KNN (K = 5) | 0.656 | 0.607 | 0.893 | 0.723 |

## How to run
```bash
git clone https://github.com/YOUR-USERNAME/week3-supervised-models.git
cd week3-supervised-models
pip install -r requirements.txt
jupyter notebook week3_supervised_models.ipynb   # then Kernel > Restart & Run All
```
Keep `urdu_news.csv` in the same folder as the notebook.

## Files
- `week3_supervised_models.ipynb` – full notebook (outputs included)
- `urdu_news.csv` – dataset
- `requirements.txt` – Python packages
