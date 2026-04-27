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
- We used `all-MiniLM-L6-v2` from Sentence Transformers to encode all training utterances into 384 dimensional vectors
- The query is formed by concatenating all conversation history utterances up to turn 5

### Preprocessing
- Data loaded from CSV files train, dev, and test
- Emotion and Empathy scores normalized to [0, 1] using min max normalization
- EmotionalPolarity one hot encoded into 4 classes (0, 1, 2, 3)
- Malformed rows are skipped

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

There were no differences in output scores of all the weights tested, so it is concluded that SBERT text similarity will dominate the retrieval regardless of what weights are given to each of the emotional components, as every combination produced the same outcome. The final weights decided upon were 0.6 for w1, 0.15 for w2, 0.15 for w3 and 0.1 for w4 because they gave the highest weighting to the text similarity which is clearly the most significant element.

### Generation
- For each conversation, the first 5 turns are used as history
- The most similar training sentence is retrieved and added to the history for each subsequent turn
- Already used sentences are masked to avoid repetition
- 5 turns (turns 6-10) are generated per conversation

## Dev Set Results (picture named `Q1_corpus_dev_metrics.PNG`)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.1279 |
| ROUGE-2 | 0.0162 |
| ROUGE-L | 0.1037 |
| BLEU | 0.0000 |
| BertScore Precision | 0.8619 |
| BertScore Recall | 0.8548 |
| BertScore F1 | 0.8582 |

The low ROUGE and BLEU scores are expected for a retrieval approach since these metrics measure exact word overlap. BertScore (0.8582) better reflects generation quality by measuring semantic similarity.

## Output
- `generations_corpus.csv` — test set generations with columns: `conversation_id, turn_number, generated_response`
- `generations_corpus_10turns.csv` — 5 dev conversations with 10 generated turns each for Q3 analysis

# Q2: In-Context learning with Large Language Models 

## Code File
`Q2_InContext_Learning_LLM.ipynb`

## How to Run
1. Open `Q2_InContext_Learning_LLM.ipynb` in Google Colab
2. Run all cells in order from top to bottom (Runtime → Run all)
3. The test generations will be saved as `q2_test_predictions.csv` automatically

## Implementation Details

### Model 
- Originally attempted to used Phi-3-Mini model, but resorted to 'microsoft/phi-2' from HuggingFace Transformers
- Loaded the mdoel with AutoModelForCausalLM and AutTokenizer
- Set the tokenizer padding token to the EOS token

### Prompt Construction 
- Each input conversation uses the frist 5 turns as context
- Few-shot examples are sampled from training set
- Each example contains:
    - Conversation History (turns 1-5)
    - Gold Conitnuation (turns 6-15)
 - Tested both 3-shot prompting and 5-shot prompting
   
### Generation
- The model was promoting to generate the next 10 turns for each conversation, starting at turn 6
- Generated outputs were parsed to extract turns 6-15
- If less than 10 turns were generated, missing turns were left as blank in CSV
- 
## Dev Set Results (picture named `q2_dev_metrics.png`)

| Metric | 3-Shot Score | 5-Shot Score
|--------|-------|-------|
| ROUGE-1 | 0.2713 |0.2495
| ROUGE-2 | 0.0472 | 0.0469
| ROUGE-L | 0.1439 | 0.1478
| BLEU | 0.0165 | 0.0088
| BertScore Precision | 0.8417 | 0.8405
| BertScore Recall | 0.812 | 0.808
| BertScore F1 | 0.8265 | 0.8238

3-Shot metrics performed better, with higher Rouge-1, Rouge-2, Bleu, and BertScore F1. Due to this, 3-shot was selected for the final test predictions.

## Output
- `q2_test_predictions.csv` — test set generations with columns: `id, turn_number, generated_response`

