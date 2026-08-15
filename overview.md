# VESTURE — Project Overview

**Virtual Clothing Try-On + Fashion Stylist AI**  
BIT4443 Deep Learning · Group Project · Streamlit Application

---

## 1. Problem Statement

Online shoppers cannot physically try outfits before buying. Fit, color, and style become guesswork, which leads to:

- Uncertain purchase decisions
- Higher product return rates
- Wasted time comparing clothes across stores

**Target users:** fashion e-commerce shoppers, stylists, and students exploring computer-vision + LLM applications.

**Motivation:** Combine deep learning (vision) with a language model (styling advice) so users can _see_ clothes on their body and _understand_ what to buy next.

---

## 2. Expected AI Solution

**VESTURE** is an intelligent Streamlit app that:

1. Uploads a full-body person photo and a garment image
2. Segments the clothing region with **SegFormer**
3. Virtually dresses the person with **IDM-VTON** (garment-image conditioned; CatVTON / SD2 fallback)
4. Recommends **Top-5 similar items** with **FashionCLIP**
5. Uses **Gemini LLM** to describe the garment, give style advice, and explain results
6. Provides **Buy / Find online** links for recommended items

Users get a visual try-on **and** a natural-language stylist — not just a raw model output.

---

## 3. Solution Steps

| Step | Action                                    | Technology              |
| ---- | ----------------------------------------- | ----------------------- |
| 1    | Upload person photo + garment             | Streamlit UI            |
| 2    | Preprocess (EXIF, RGB, resize)            | Pillow / OpenCV         |
| 3    | Segment clothing region + `seg_conf`      | SegFormer (HF)          |
| 4    | Gate: require confidence ≥ **0.85**       | Custom scoring          |
| 5    | Caption garment (color, fabric, style)    | Gemini Vision API       |
| 6    | Dress person with the garment image       | IDM-VTON Space (CatVTON / SD2 fallback) |
| 7    | Score try-on quality                      | Composite confidence    |
| 8    | Retrieve Top-5 similar catalog items      | FashionCLIP             |
| 9    | Write stylist advice + result explanation | Gemini Text API         |
| 10   | Shop similar items                        | Buy / Find online links |

---

## 4. Workflow

### 4.1 High-level pipeline

```text
Person photo ──► SegFormer ──► mask + seg_conf
                                      │
                                      ▼
                              seg_conf ≥ 0.85?
                                      │ yes
Garment image ──► Gemini Vision ──► rich garment_des
                                      │
                                      ▼
                    IDM-VTON Space (person + garment)
                         CatVTON / SD2 fallback
                                      │
                                      ▼
                               Try-on result
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              FashionCLIP      Try-on confidence    Gemini Stylist
              Top-5 + shop      (composite score)   advice + explain
```

### 4.2 Deep Learning path (perception)

1. **SegFormer** finds where clothes are on the body
2. **IDM-VTON** transfers the garment *image* onto the body (CatVTON / SD2 if the Space is busy)
3. **FashionCLIP** finds visually similar products

### 4.3 LLM path (language / styling)

1. **Gemini Vision** describes the garment → IDM-VTON `garment_des` / SD prompt
2. **Gemini Text** gives occasion/season pairing advice
3. **Gemini Text** explains confidence scores and Top-5 in plain English

### 4.4 Gemini adaptation (not full fine-tuning)

We use **pretrained Gemini via API inference**, adapted with:

- Fashion-stylist system prompts
- Few-shot examples
- Multimodal garment image input
- Grounding on SegFormer / FashionCLIP scores

**Viva answer:** _We did not fine-tune Gemini weights. We adapted pretrained Gemini through inference + domain prompting, grounded on our deep-learning outputs — allowed under “transfer learning, fine-tuning, or model inference.”_

---

## 5. Outcome

| Outcome                   | Description                                                                 |
| ------------------------- | --------------------------------------------------------------------------- |
| **Visual try-on**         | Before/after image of the user “wearing” the garment                        |
| **Confidence scores**     | Segmentation, CLIP similarity, mask quality, try-on composite (≥ 0.85 gate) |
| **Top-5 recommendations** | Similar catalog items with similarity %                                     |
| **Shop actions**          | Buy / Find online buttons per item                                          |
| **AI stylist copy**       | Garment description + advice + explanation (with Gemini key)                |
| **Demo-ready UI**         | Polished Streamlit app (VESTURE brand)                                      |

**End-to-end experience:** upload → dress → explain → shop.

---

## 6. Tech Stack

| Layer            | Technology                                            |
| ---------------- | ----------------------------------------------------- |
| UI               | Streamlit                                             |
| Language         | Python 3.10+                                          |
| Deep Learning    | PyTorch, Hugging Face Transformers                    |
| Vision models    | SegFormer, IDM-VTON, CatVTON, SD2 Inpainting, FashionCLIP |
| LLM              | Google Gemini API (`GOOGLE_API_KEY`)                  |
| Cloud inference  | Hugging Face Spaces + Inference API (`HF_TOKEN`)      |
| Image processing | Pillow, OpenCV, scikit-image, SciPy                   |
| Data             | Pandas, NumPy                                         |
| Config           | python-dotenv (`.env`)                                |

### API keys

| Key              | Purpose                          | Required?                               |
| ---------------- | -------------------------------- | --------------------------------------- |
| `HF_TOKEN`       | IDM-VTON / CatVTON Spaces + SD2 fallback | Yes for live try-on                     |
| `GOOGLE_API_KEY` | Gemini stylist / garment caption | Optional (fallback to template prompts) |

**Note:** No OpenAI key required. Gemini is the chosen LLM.

---

## 7. Models

| Model                             | ID / Source                                 | Role                               | Runs on          |
| --------------------------------- | ------------------------------------------- | ---------------------------------- | ---------------- |
| **SegFormer-B2 Clothes**          | `mattmdjaga/segformer_b2_clothes`           | Clothing segmentation + `seg_conf` | Local CPU        |
| **IDM-VTON**                      | `yisol/IDM-VTON` Space                      | Garment-image virtual try-on       | HF Space GPU     |
| **CatVTON**                       | `zhengchong/CatVTON` Space                  | Try-on fallback                    | HF Space GPU     |
| **Stable Diffusion 2 Inpainting** | `stabilityai/stable-diffusion-2-inpainting` | Last-resort text-guided try-on     | HF Inference API |
| **Marqo FashionCLIP**             | `Marqo/marqo-fashionCLIP`                   | Top-5 similar items                | Local CPU        |
| **Gemini**                        | Google Generative AI API                    | Caption, advice, explanation       | Cloud API        |
| Fallback CLIP                     | `openai/clip-vit-base-patch32`              | If FashionCLIP fails               | Local CPU        |

### Why these models

- **SegFormer:** Fashion-specific labels (upper-clothes, pants, dress); strong Acc/IoU; CPU-friendly
- **IDM-VTON:** Garment-*image* conditioned; far stronger pattern transfer than text inpaint
- **CatVTON / SD2:** Fallbacks when the IDM-VTON Space is queued
- **FashionCLIP:** Outperforms generic CLIP on fashion retrieval
- **Gemini:** Multimodal (image + text); student-friendly API; stylist language layer

### SegFormer published metrics (cite in slides)

- Mean Accuracy **0.80**, Mean IoU **0.69**
- Upper-clothes Acc **0.87** / IoU **0.78**
- Pants Acc **0.90** / IoU **0.84**

### Confidence formulas

```text
seg_conf   = mean softmax over selected clothing pixels
tryon_conf = 0.4·seg_conf + 0.4·CLIP_sim + 0.2·mask_quality
reco_conf  = cosine similarity (FashionCLIP); highlight ≥ 0.85
```

Hard gate: try-on blocked if `seg_conf < 0.85`.

---

## 8. Datasets

| Dataset                             | Stats                                             | Role in project                        |
| ----------------------------------- | ------------------------------------------------- | -------------------------------------- |
| **ATR / human_parsing_dataset**     | 18 parsing classes                                | SegFormer training provenance          |
| **VITON-HD**                        | 13,679 pairs; 1024×768; train 11,647 / test 2,032 | Try-on benchmark + optional Studio demos |
| **Dress Code**                      | 53,792 garments; 107,584 images                   | Multi-category (upper / lower / dress)   |
| **DeepFashion / fashion products**  | ~44k–52k catalog images                           | Runtime Top-5 catalog                    |
| **Local catalog** (`data/catalog/`) | ~80–150 real product images                       | Runtime Top-5 in the app                 |
| **Self-collected photos**           | 20–50 recommended                                 | Live demo + confidence analysis        |

### Dataset usage summary

| Dataset        | Train from scratch?       | App runtime?      | Slides? |
| -------------- | ------------------------- | ----------------- | ------- |
| ATR            | No (already in SegFormer) | Via model weights | Yes     |
| VITON-HD       | No                        | Optional demos    | Yes     |
| Dress Code     | No                        | Optional          | Yes     |
| DeepFashion    | No                        | **Yes — catalog** | Yes     |
| Local catalog  | No                        | **Yes**           | Yes     |
| Self-collected | No                        | **Yes**           | Yes     |

### Data preprocessing

1. Person: EXIF → RGB → letterbox resize (768×1024 or fast 384×512)
2. Mask: SegFormer → class select → dilate → feather
3. Garment: RGB → resize → Gemini caption (or template prompt)
4. Catalog: clean titles → FashionCLIP embed → `embeddings.npy` + `catalog.csv`
5. Gate: reject / warn if `seg_conf < 0.85`

---

## 9. Streamlit Application Features

- Project title: **VESTURE**
- Tabs: **Home** · **Studio** · **Catalog** · **Stylist AI** · **Profile**
- Upload person + garment, or pick a VITON-HD demo pair
- Prediction button: **Generate Try-On**
- Results: before/after, mask, confidence metrics
- Top-5 similar items + Buy / Find online
- LLM panels: AI garment description, stylist advice, result explanation

---

## 10. Project Structure

```text
Virtual_Try_On/
├── app.py                 # Streamlit UI (VESTURE)
├── overview.md            # This document
├── README.md
├── requirements.txt
├── .env.example           # HF_TOKEN, GOOGLE_API_KEY
├── src/
│   ├── preprocess.py
│   ├── segmentation.py    # SegFormer
│   ├── tryon.py           # IDM-VTON → CatVTON → SD2
│   ├── confidence.py
│   ├── recommend.py       # FashionCLIP Top-5
│   ├── catalog_builder.py
│   ├── llm_advisor.py     # Gemini caption / advice / explain
│   └── demo_samples.py    # VITON-HD-style Studio pairs
├── data/catalog/
├── data/samples/
├── assets/home/
└── assets/demo/           # Offline try-on fallback images
```

---

## 11. Strengths, Limitations, Future Work

### Strengths

- Pretrained HF models only (no training from scratch)
- Works without a local GPU (IDM-VTON / CatVTON on HF Spaces)
- Clear confidence gate (≥ 0.85)
- DL + LLM hybrid story for presentation
- Polished Streamlit demo UI

### Limitations

- Space queues / API latency (HF + Gemini)
- Fails on extreme poses, occlusion, multi-person photos
- Try-on confidence is a **proxy score**, not a calibrated probability
- IDM-VTON / CatVTON are CC-BY-NC-SA 4.0 (class demo, not commercial)

### Future improvements

- Larger real DeepFashion catalog
- Optional real Gemini fine-tune on fashion Q&A (Vertex) if budget/time allow

---

## 12. Course Alignment (BIT4443)

| Requirement          | How VESTURE meets it                                    |
| -------------------- | ------------------------------------------------------- |
| Real-world problem   | Online try-before-you-buy / returns                     |
| Pretrained DL models | SegFormer, IDM-VTON, FashionCLIP, Gemini            |
| Transfer / inference | HF + Gemini API inference; prompt adaptation            |
| Dataset description  | ATR, VITON-HD, Dress Code, DeepFashion, local catalog   |
| Streamlit UI         | Title, description, upload, predict, result, confidence |
| Analysis             | Confidence metrics, strengths/limits                    |

---
