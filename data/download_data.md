# Getting the dataset

The raw `creditcard.csv` (~98 MB, 284,807 rows) is not committed to this repo — GitHub isn't a great place to store large binary/CSV data files, and the dataset is freely available directly from its source.

## Option 1 — Kaggle web UI (easiest)

1. Go to [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Sign in (free account)
3. Click **Download** and unzip
4. Place `creditcard.csv` in this `data/` folder

## Option 2 — Kaggle CLI

```bash
pip install kaggle
# place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

Once `data/creditcard.csv` is in place, run `python src/train.py` from the repo root.
