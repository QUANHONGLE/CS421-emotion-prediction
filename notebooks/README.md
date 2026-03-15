# Q1: ANN with Vector Embeddings

## Code File
`Q1_ANN_Embeddings.ipynb`

## How to Run
1. Open `Q1_ANN_Embeddings.ipynb` in Google Colab
2. Run all cells in order from top to bottom (Runtime -> Run all)
3. The predictions will be saved as `predictions_ann.csv` after running

## Implementation Details

### Embeddings
Two models were used to convert each conversational turn into a fixed-length vector:
- **GloVe** (`glove-wiki-gigaword-100`) — loaded using Gensim. Each word in a sentence is converted to a 100 dimensional vector, then averaged to get one vector per sentence
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — encodes the full sentence directly into a 384 dimensional vector

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
# Q2: Fine-tuning an RNN-based Model

## Code File
`Q2_Fine_Tuning_RNN_based_Model.ipynb`

## How to Run
1. Open `Q2_Fine_Tuning_RNN_based_Model.ipynb` in Google Colab
2. Run all cells in order from top to bottom (Runtime -> Run all)
3. The predictions will be saved as `predictions_rnn.csv` after running

## Implementation Details

### Embeddings
A tokenizer-based apporach was used for the RNN where the data is numerized and padding to a fixed length 

### Model Architecture
A recurrent nueral network (RNN) was built in PyTorch with the following structuere:
- Single-layer LTSM
- Embedding layer converts the numerized padded tokens into dense vectors
- Dropout rate of 0.3 applied to the final state before classification
- Three task-specific output heads on the shared hidden state:
  - Emotion (regression, outputs 1 value)
  - Emotional Polarity (classification, outputs 4 class scores)
  - Empathy (regression, outputs 1 value) 

### Loss Functions
- MSE Loss for Emotion and Empathy (regression tasks)
- Cross Entropy Loss for Emotional Polarity (classification task)

### Training
- Optimizer: Adam (learning rate = 0.001)
- Batch size: 32
- Epochs: 10
- Trained on GPU (Google Colab T4)

### Preprocessing
- Data loaded from CSV files (train, dev, test)
- Train text and dev text data was stripped of punctuation and placed into lowercase and returned as a list
- Cleaned train text and dev text data was tokenized
- Tokenized train text data was used to build a dictionary of vocabulary with an index, this included <PAD> and <UNK> symbols as well
- Train, dev, and test data was normalized using the pre-built vocabulary dictionary
- For loop used to get the length of each sentence in the train text to be used for padding
- Using np.percentile determined that the 95% would be the appropriate length for the padding, length resulted in 45
- Function created to iterate through each data set and ensure that each sentence was of length 45 

## Normalized text result 
[[216, 192, 250, 119, 5084, 692, 167, 433, 52, 38, 124, 21, 101, 115, 2, 13, 5, 45, 226, 6, 7, 8],… ] 


## Model Choice for Final Predictions
LSTM model was used for the final predictions since it is better at handling long term dependencies. This is vital for detecting emotions where context plays a significant role to the meaning. 

## Output
Predictions saved to `predictions_rnn.csv` with columns: `id, Emotion, EmotionalPolarity, and Empathy`

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
- Optimizer: AdamW
- Scheduler: Linear warmup schedule
- Batch size: 16
- Max sequence length: 128
- Epochs: 3
- Trained on T4 GPU

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
# Q4: LLM Prompting

## Code File
`Q4_Prompting_with_LLMs.ipynb`

## How to Run
1. Open `Q4_Prompting_with_LLMs.ipynb` in Google Colab
2. Make sure GPU is enabled: Runtime -> Change runtime type -> T4 GPU
3. Run all cells in order from top to bottom (Runtime -> Run all)
4. The 5 conversations will be exported to 5_convos.csv

## Implementation
-  Utilzing pandas conversations were grouped by conversation_id and counts were taken
-  Afterwards, conversations with 10 or more utterances were filtered out and the first 5 conversational id's were extracted
-  Conversation id's and texts were extracted into CSV file to be uploaded to a LLM for processing
-  Analysis was then saved into LLM_output.txt file 

## Model Selection 
- Claude (claude-sonnet-4-20250514) accessed via the Anthropic API since it was able to apply different prompting techniques (zero-shot, few-shot, and chain-of-thought) for the analysis
- Analysis displayed and emotion intensity score per speaker, empathy score for the overall conversation, sentiment polarity classification, and top 3 dominant emotions

