# Q3: Fine-tuning a Transformer-based Model

## Code File
`Q3_BERT_Finetuning.ipynb`

## How to Run
1. Open `Q3_BERT_Finetuning.ipynb` in Google Colab
2. Make sure GPU is enabled: Runtime → Change runtime type → T4 GPU
3. Run all cells in order from top to bottom (Runtime → Run all)
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

---

# Q1: ANN with Vector Embeddings

## Code File
`Q1_ANN_Embeddings.ipynb`

## How to Run
1. Open `Q1_ANN_Embeddings.ipynb` in Google Colab
2. Run all cells in order from top to bottom (Runtime → Run all)
3. The predictions will be saved as `predictions_ann.csv` automatically

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

## Dev Set Results

| Model | MAE Emotion | MAE Empathy | F1 Polarity |
|-------|------------|-------------|-------------|
| GloVe | 0.525 | 0.766 | 0.492 |
| SBERT | 0.479 | 0.741 | 0.630 |

## Model Choice for Final Predictions
SBERT was used for final test predictions as it outperformed GloVe across all three metrics, likely because it encodes full sentence meaning rather than averaging individual word vectors.

## Output
Predictions saved to `predictions_ann.csv` with columns: `id, Emotion, EmotionalPolarity, Empathy`

---

# Project Part 2

# Q1: Corpus-Based Chatbot

## Code File
`Q1_Corpus_Chatbot.ipynb`

## How to Run
1. Open `Q1_Corpus_Chatbot.ipynb` in Google Colab
2. Run all cells in order from top to bottom (Runtime → Run all)
3. The test generations will be saved as `generations_corpus.csv` automatically

## Implementation Details

### Embeddings
- Used `all-MiniLM-L6-v2` from Sentence Transformers to encode all training utterances into 384-dimensional vectors
- The query is formed by concatenating all conversation history utterances up to turn 5

### Preprocessing
- Data loaded from CSV files (train, dev, test)
- Emotion and Empathy scores normalized to [0, 1] using min-max normalization
- EmotionalPolarity one-hot encoded into 4 classes (0, 1, 2, 3)
- Malformed rows skipped using `on_bad_lines='skip'`

### Similarity Score
The total similarity between a query and each training utterance is calculated as:

Stotal = w1·stext + w2·semotion + w3·sempathy + w4·spolarity

Where:
- stext = cosine similarity between SBERT embeddings
- semotion = 1 - |EI1 - EI2| (emotion intensity similarity)
- sempathy = 1 - |Emp1 - Emp2| (empathy similarity)
- spolarity = 1 if polarities match, 0 otherwise

### Weight Selection
Four weight configurations were tested on a subset of 20 dev conversations:

| w1 (text) | w2 (emotion) | w3 (empathy) | w4 (polarity) | ROUGE-1 | BertScore F1 |
|-----------|-------------|-------------|--------------|---------|-------------|
| 0.60 | 0.15 | 0.15 | 0.10 | 0.1364 | 0.8592 |
| 0.25 | 0.25 | 0.25 | 0.25 | 0.1364 | 0.8592 |
| 0.70 | 0.10 | 0.10 | 0.10 | 0.1364 | 0.8592 |
| 0.40 | 0.20 | 0.20 | 0.20 | 0.1364 | 0.8592 |

All weight configurations produced identical scores, indicating that SBERT text similarity dominates the retrieval process regardless of the emotional component weights. The final weights chosen were w1=0.6, w2=0.15, w3=0.15, w4=0.1 as they give the most weight to text similarity which is clearly the most impactful component.

### Generation
- For each conversation, the first 5 turns are used as history
- The most similar training sentence is retrieved and added to the history for each subsequent turn
- Already used sentences are masked to avoid repetition
- 5 turns (turns 6-10) are generated per conversation

## Dev Set Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.1279 |
| ROUGE-2 | 0.0162 |
| ROUGE-L | 0.1037 |
| BLEU | 0.0000 |
| BertScore Precision | 0.8619 |
| BertScore Recall | 0.8548 |
| BertScore F1 | 0.8582 |

The low ROUGE and BLEU scores are expected for a retrieval-based approach since these metrics measure exact word overlap. BertScore (0.8582) better reflects generation quality by measuring semantic similarity.

## Output
- `generations_corpus.csv` — test set generations with columns: `conversation_id, turn_number, generated_response`
- `generations_corpus_10turns.csv` — 5 dev conversations with 10 generated turns each for Q3 analysis
