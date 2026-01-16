# Project Report: Hindi Chunking and Analysis

## Part 1: Transformers for Sequence Classification
In this section, token classification was implemented using DistilBERT on the WNUT 17 dataset.

**Challenges & Solutions:**
- **Dataset Loading**: The `wnut_17` dataset on Hugging Face relies on a loading script that is no longer supported for security reasons (`trust_remote_code` is deprecated).
- **Workaround**: A manual loading function was implemented using `requests` to fetch the raw data from the official GitHub repository, which was then parsed locally. This ensured the correct data was used, replacing the standard Hugging Face loading script with custom parsing logic.

## Part 2 & 3: Hindi Chunking

### Methodology
The objective was to perform Chunking on a Hindi CoNLL-U dataset.
- **Task**: Identify chunks (e.g., Noun Phrases - NP) and their types (Head/Child).
- **Label Cleaning**: Trailing numbers were removed from ChunkIDs (e.g., `NP2` became `NP`) to avoid label sparsity and focus on the linguistic category rather than the specific instance index.

**Bonus Task: Joint Classification Strategy**
Instead of training two separate models (one for ChunkID, one for ChunkType), a **Joint Classification** strategy was implemented, combining labels into a single schema: `IOB-ChunkID-ChunkType` (e.g., `B-NP-head`).

*Rationale for Joint Classification:*
Depending on the task complexity, separate or joint classification might be preferred. Here, the joint approach was chosen for several reasons:
1.  **Error Propagation**: In a pipelined approach (where a Chunk classifier feeds into a Type classifier), mistakes made by the first model create a flawed starting point for the second. A joint model avoids this cascading error effect.
2.  **Performance Benchmark**: For a separate classifier approach to be viable, the combined accuracy of both models would need to significantly outperform the single joint model. Given the high performance achieved by the joint model, separate classifiers would likely introduce unnecessary computational overhead and complexity without a guaranteed performance gain.
3.  **Mutual Information**: It effectively allows the model to learn the structural dependencies between chunk boundaries and their internal roles (Head/Child) simultaneously.

### Model Comparison
Three models were compared to understand how pre-training domain affects performance on Hindi:

1.  **`distilbert-base-multilingual-cased`** (Multilingual Baseline):
    -   *Motivation*: This model was pre-trained on 104 languages, including Hindi. It served as the primary baseline, expected to perform reasonably well due to its exposure to Hindi text.

2.  **`distilbert-base-uncased`** (English Baseline):
    -   *Motivation*: This model was pre-trained **only** on English. It was included as a negative control. It was expected to perform poorly because it has likely never seen Hindi characters and its tokenizer would presumably fragment Hindi words into many unknown tokens, degrading performance significantly.

3.  **`mirfan899/hindi-distilbert-ner`** (Fine-tuned Hindi Model):
    -   *Motivation*: This is a version of DistilBERT that has ostensibly been fine-tuned on Hindi NER tasks. It was included to see if transfer learning from a related Hindi task (NER) would provide better initialization for Chunking than the raw multilingual model.

### Results
| Model | Precision | Recall | F1 Score | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Multilingual Baseline** | 0.965 | 0.962 | 0.964 | 0.965 |
| **English Baseline** | 0.866 | 0.860 | 0.863 | 0.870 |
| **Fine-tuned Hindi** | 0.966 | 0.963 | 0.965 | 0.965 |

### Analysis
- **Multilingual Model**: As expected, performance was excellent (~96% F1) due to pre-training on Hindi data.
- **English Model**: Surprisingly, the English model achieved a high F1 score (~86%).
    - **Investigation**: It was observed that the English tokenizer falls back to **character-level tokens** for Hindi (e.g., `['य', '##ह', ...]`) instead of `[UNK]`.
    - **Alignment Strategy**: Transformers use subword tokenization (WordPiece), meaning one word might be split into multiple tokens. The single label provided for the word (e.g., `B-NP`) must be aligned to these multiple tokens. Following the tutorial strategy, the label is assigned to the **first subword**, and subsequent subwords are ignored (set to `-100`).
    - **Impact**: Because of this strategy, the English model might display worse metrics when evaluated against a bigger or more varied dataset. It might have learned some common structures from the available datasets, but real-world data (or a larger volume of data) might not be as easy to predict based only on the first subword. A different strategy would likely need to be introduced for robust performance. Even with this dataset, further evaluation and comparison of different approaches to this alignment problem might have provided better or more reliable results.
    - **Conclusion**: The current success suggests the model learned the **structure** of the IOB tags and phrase boundaries based purely on character patterns and sentence position, acting effectively as a character-level model, despite having no semantic understanding of Hindi words.
- **Fine-tuned Hindi Model**: Performed on par with the Multilingual Baseline (difference was negligible, ~0.1% improvement).
    - **Observation**: Transfer learning from a related task (NER) did not yield significant improvements over the strong multilingual baseline for this specific chunking task. This might be because the domain shift between the NER dataset and this Chunking dataset is too large, or simply that the Multilingual BERT is already optimized enough for Hindi structure.

### Conclusions
1.  **Multilingual BERT is Sufficient**: For Hindi Chunking, the standard `distilbert-base-multilingual-cased` is highly effective (96.4% F1), offering a remarkable level of performance without further pre-training.
2.  **Structure vs. Semantics**: The surprising performance of the English model highlights that Chunking is a highly **structural** task. A model can go a long way just by learning character patterns and label transitions, even without understanding the meaning of words.
3.  **Joint Classification Works**: The bonus implementation of combining ChunkID and ChunkType (e.g. `NP-head`) was successful, allowing a single model to solve the full task structure with high accuracy.

## Acknowledgments
- **Tutorial Reference**: The workflow in Part 1 follows the [Hugging Face Token Classification Tutorial](https://huggingface.co/docs/transformers/tasks/token_classification).
- **Dataset Credit**: The dataset was provided by the university server (Hindi HDTB-UD).
- **Model Credit**: The fine-tuned Hindi model used in Experiment 3 was provided by `mirfan899` on Hugging Face.
