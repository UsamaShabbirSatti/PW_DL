├── Copy_training.json
├── Copy_validation.json
└── copy_test.json
The notebook will:

Load all three splits as Pandas DataFrames
Apply majority voting across six annotators to assign a single label per tweet (items without a clear majority are dropped)
Filter to English-only tweets (lang == 'en')
Clean tweets by removing emojis, hashtags, mentions, URLs, and special characters, followed by lemmatization
Encode text using GloVe embeddings, with OOV tokens handled via a static <UNK> embedding


3. Model Training
Two Bi-LSTM architectures and one Transformer model are trained and compared.
Bi-LSTM models (trained with 3 seeds: 42, 111, 321 for robust estimation):

Baseline — single Bidirectional LSTM + Dense output layer
Stacked — two Bidirectional LSTM layers + Dense output layer

Transformer model:

cardiffnlp/twitter-roberta-base-hate fine-tuned using HuggingFace Trainer

The best checkpoint per run is saved to Drive via ModelCheckpoint, monitored on val_f1 (macro F1-score).
