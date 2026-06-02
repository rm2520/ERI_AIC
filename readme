# Arabic Image Captioning using Deep Learning

An Arabic Image Captioning framework that combines Computer Vision and Natural Language Processing to generate meaningful Arabic descriptions from images.

This project benchmarks multiple image captioning architectures on three Arabic captioning datasets:

- ERI-AIC (Proposed Dataset)
- Arabic Flickr8k
- Arabic MS-COCO

## ?? Model Architectures

The framework evaluates five state-of-the-art image captioning architectures:

1.Encoder–Decoder (VGG19 + LSTM):** Uses a pre-trained VGG19 network to extract deep visual features, paired with a sequence-generating Long Short-Term Memory (LSTM) decoder network.
2.MobileNetV2 + LSTM + Attention:** Combines a lightweight, efficient MobileNetV2 feature extractor with an LSTM decoder utilizing a spatial attention layer to focus on relevant image regions dynamically.
3.MobileNetV2 + GRU + Attention:** Swaps the sequential decoder for a Gated Recurrent Unit (GRU) to optimize computational efficiency while keeping attention layers intact.
4.Top-Down Attention Model:** Employs a dual-LSTM network design. An attention LSTM determines where to focus visual field dependencies, while a language generation LSTM predicts output text sequences.

### Key Characteristics

- 4,000 images
- 12,000 human-authored Arabic captions
- 3 captions per image
- Captions written directly in Arabic (not translated)
- Accessibility-oriented descriptions
- Culturally grounded Arab scenes
- Urban and rural environments

### Dataset Design Goals

- Support Arabic image captioning research
- Improve caption quality for Arabic language generation
- Provide richer contextual information
- Better serve visually impaired users

---

## Dataset Comparison

| Dataset | Images | Captions/Image | Caption Source |
|----------|---------|----------------|----------------|
| Arabic Flickr8k | 8,000 | 3 | Translated |
| Arabic MS-COCO | 82,000+ | 3 | Translated |
| ERI-AIC | 4,000 | 3 | Human-written Arabic |

---

## Preprocessing Pipeline

### Image Processing

- Resize to 224 × 224
- RGB channels
- Pixel normalization to [-1, 1]

### Arabic Text Processing

- Diacritics removal
- AraBERT word segmentation
- Punctuation removal
- Tokenization
- Vocabulary creation
- `<start>` and `<end>` sequence tokens



## Project Structure

```text
.
+-- data/
¦   +-- ERI-AIC/
        +--ERI_Dataset
		+--Wasef_text
		   +--final_text_caption
		   +--wasef_train80
		   +--wasef_test20
¦   +-- Flickr8k/
¦   +-- MSCOCO/
¦
+-- models/
¦   +-- encoder_decoder/
¦   +-- lstm_attention/
¦   +-- gru_attention/
¦   +-- top_down_attention/


+-- README.md
```

---

## Experimental Results

### Arabic Flickr8k

| Model | BLEU-1 | BLEU-4 | METEOR | CIDEr | ROUGE |
|---------|---------|---------|---------|---------|---------|
| Encoder–Decoder | 32.9 | 4.2 | 0.246 | 0.225 | 0.14 |
| LSTM + Attention | 48.5 | 11.6 | 0.350 | 0.310 | 0.32 |
| GRU + Attention | 48.7 | 11.3 | 0.348 | 0.316 | 0.31 |
| Top-Down Attention | 45.4 | 9.2 | 0.317 | 0.255 | 0.29 |


---

### Arabic MS-COCO

| Model | BLEU-1 | BLEU-4 | METEOR | CIDEr | ROUGE |
|---------|---------|---------|---------|---------|---------|
| Encoder–Decoder | **61.9** | **42.1** | **0.720** | 0.265 | 0.10 |
| LSTM + Attention | 40.9 | 6.8 | 0.289 | 0.200 | 0.30 |
| GRU + Attention | 41.2 | 7.0 | 0.290 | 0.210 | 0.31 |
| Top-Down Attention | 37.2 | 4.9 | 0.250 | 0.160 | 0.24 |


---

### ERI-AIC (Proposed Dataset)

| Model | BLEU-1 | BLEU-4 | METEOR | CIDEr | ROUGE |
|---------|---------|---------|---------|---------|---------|
| Encoder–Decoder | 41.0 | 9.6 | 0.255 | 0.265 | 0.15 |
| LSTM + Attention | 64.4 | 20.9 | 0.401 | 0.338 | 0.35 |
| GRU + Attention | 64.1 | 21.3 | 0.412 | 0.376 | 0.36 |
| Top-Down Attention | 61.3 | 15.1 | 0.319 | 0.270 | 0.32 |

---

## Key Findings

- Original Arabic captions outperform translated datasets.
- ERI-AIC consistently improves caption quality across all architectures.
- Transformer models achieve the best overall performance.
- ERI-AIC captions are:
  - Longer
  - More informative
  - More diverse
  - More culturally grounded

---

## Cross-Dataset Generalization

Models trained on ERI-AIC demonstrated:

? Better semantic alignment

? Improved contextual understanding

? Fewer hallucinated objects

? Stronger robustness on unseen scenes

Compared with models trained on Arabic Flickr8k and Arabic MS-COCO.

---
## Download Evaluation Toolkit

This project uses the COCO caption evaluation metrics (BLEU, METEOR, ROUGE-L, CIDEr) through pycocoevalcap.

install the evaluation package:

pip install pycocoevalcap

## Download dataset 

## Installation

```bash
git clone https://github.com/rm2520/Image-caption_model1.git

cd Image-caption_model1


```

---

## Training
Model configs

- embedding_dim: 512
- max_seq_len: 128
- dropout: 0.5

###  Hyperparameters

- n_epochs: 100
- batch_size: 100
- hidden_layer :512
- learning_rate: 1e-2
- optimizer: Adam
- loss: CrossEntropyLoss
- early_stopping: 30

```bash
python main.py
```

