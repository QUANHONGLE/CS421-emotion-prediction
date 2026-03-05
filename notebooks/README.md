# Q1: ANN with Vector Embeddings

## Code File
`Q1_ANN_Embeddings.ipynb`

## How to Run
1. Open `Q1_ANN_Embeddings.ipynb` in Google Colab
2. Run all cells in order from top to bottom (Runtime → Run all)
3. The predictions will be saved as `predictions_ann.csv` after running

## Implementation Details

### Embeddings
Two pretrained embedding models were used to convert each conversational turn into a fixed-length vector:
- **GloVe** (`glove-wiki-gigaword-100`) — loaded via Gensim. Each word in a sentence is converted to a 100-dimensional vector, then averaged to get one vector per sentence
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — encodes the full sentence directly into a 384-dimensional vector

### Model Architecture
A feedforward ANN was built in PyTorch with the following structure:
- Two shared fully connected layers with ReLU activation and 0.3 dropout
- Three task-specific output heads:
  - Emotion head (regression, outputs 1 value)
  - EmotionalPolarity head (classification, outputs 4 class scores)
  - Empathy head (regression, outputs 1 value)

### Loss Functions
- MSE Loss for Emotion and Empathy (regression tasks)
- Cross Entropy Loss for EmotionalPolarity (classification task)

### Training
- Optimizer: Adam (learning rate = 0.001)
- Batch size: 32
- Epochs: 10
- Trained on CPU

### Preprocessing
- Data loaded from CSV files (train, dev, test)
- Malformed rows in test set skipped using `on_bad_lines='skip'`
- EmotionalPolarity values are 0-indexed (0, 1, 2, 3) — no shifting needed
- RESULTS WILL VARY AFTER EVERY RERUN

## Dev Set Results

| Model | MAE Emotion | MAE Empathy | F1 Polarity |
|-------|------------|-------------|-------------|
| GloVe | 0.556 | 0.761 | 0.521 |
| SBERT | 0.506 | 0.740 | 0.628 |

## Model Choice for Final Predictions
SBERT was used for final test predictions as it outperformed GloVe across all three metrics, likely because it encodes full sentence meaning rather than averaging individual word vectors.

## Output
Predictions saved to `predictions_ann.csv` with columns: `id, Emotion, EmotionalPolarity, Empathy`

---

# Q3: Fine-tuning a Transformer-based Model

## Code File
`Q3_BERT_Finetuning.ipynb`

## How to Run
1. Open `Q3_BERT_Finetuning.ipynb` in Google Colab
2. Make sure GPU is enabled: Runtime -> Change runtime type -> T4 GPU
3. Run all cells in order from top to bottom (Runtime -> Run all)
4. The predictions will be saved as `predictions_bert.csv` automatically

## Implementation Details

### Pretrained Model
- **Model**: `bert-base-uncased` loaded from HuggingFace
- **Tokenizer**: `AutoTokenizer` from HuggingFace with max sequence length of 128

### Model Architecture
BERT's CLS token output (768-dimensional) is passed through dropout and into three task-specific heads:
- Emotion head (regression, outputs 1 value)
- EmotionalPolarity head (classification, outputs 4 class scores)
- Empathy head (regression, outputs 1 value)

### Loss Functions
- MSE Loss for Emotion and Empathy (regression tasks)
- Cross Entropy Loss for EmotionalPolarity (classification task)

### Training
- Optimizer: AdamW (learning rate = 2e-5)
- Scheduler: Linear warmup schedule
- Batch size: 16
- Max sequence length: 128
- Epochs: 3
- Trained on GPU (T4)

### Preprocessing
- Data loaded from CSV files (train, dev, test)
- Malformed rows in test set skipped using `on_bad_lines='skip'`
- Text tokenized using BERT tokenizer with padding and truncation
- EmotionalPolarity values are 0-indexed (0, 1, 2, 3) — no shifting needed

## Dev Set Results

| Metric | Score |
|--------|-------|
| MAE Emotion | 0.494 |
| MAE Empathy | 0.705 |
| F1 Polarity | 0.697 |

## Output
Predictions saved to `predictions_bert.csv` with columns: `id, Emotion, EmotionalPolarity, Empathy`
