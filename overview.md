# VESTURE — Project Overview

**Virtual Clothing Try-On + Fashion Stylist AI**  
BIT4443 Deep Learning · Group Project · Streamlit Application

Full model map (which model for which job, links, fine-tuning, system prompts, CV vs NLP, try-on masks): **[`MODELS.md`](MODELS.md)**.

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
6. Opens **Google Shopping** (`tbm=shop`) so the shopper can buy a similar item

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
| 10   | Shop similar items                        | **Google Shopping** (`tbm=shop`) Buy / Find online |

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
4. **Gemini Chat** (optional photos + voice) ranks catalog pieces; **Buy** opens Google Shopping

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
| **Shop actions**          | Each match opens a **Google Shopping** search for that product title        |
| **AI stylist copy**       | Garment description + advice + explanation (with Gemini key)                |
| **Demo-ready UI**         | Polished Streamlit app (VESTURE brand)                                      |

**End-to-end experience:** upload → dress → explain → shop.

---

## 6. Tech Stack

Each row is **what we use** and **which part of the app it serves**.

| Technology | Used for which part |
| --- | --- |
| **Python 3.10+** | Whole project language (`app.py`, `src/`) |
| **Streamlit** | UI: Home, Studio, Catalog, Stylist AI, Profile; upload, Generate Try-On, score bars, chat |
| **PyTorch** | Local tensors: SegFormer forward pass, FashionCLIP / CLIP encode |
| **Hugging Face Transformers** | Load SegFormer (`SegformerForSemanticSegmentation`) and CLIP fallback |
| **open-clip-torch** | Load **Marqo FashionCLIP** from the Hub for Top-5 + CLIP similarity |
| **Pillow / OpenCV** | EXIF, RGB, letterbox resize (person 768×1024, garment 512×512), JPEG catalog stills |
| **SciPy `ndimage`** | Mask **dilate**, Gaussian **feather**, connected-component **mask_quality** |
| **NumPy / Pandas** | `embeddings.npy`, `catalog.csv`, cosine `S @ q` |
| **gradio_client** | Call **IDM-VTON** and **CatVTON** Hugging Face Spaces (Studio try-on) |
| **huggingface_hub / Inference API** | Download weights; **SD 2 Inpainting** last-resort try-on |
| **google-genai** | **Gemini 2.0 Flash**: garment caption, advice, explain, avatar JSON, chat, voice STT |
| **python-dotenv** | Load `HF_TOKEN`, `GOOGLE_API_KEY`, `CONFIDENCE_GATE` from `.env` |
| **requests / datasets** | Build catalog from HF Dataset Viewer / DeepFashion-style stills |
| **Google Shopping URL** | Catalog / Top-5 **Buy / Find online** (`tbm=shop`) |

### 6.1 Stack by pipeline stage

| App part | What happens | Stack |
| --- | --- | --- |
| **Studio — upload / preprocess** | Person + garment photos | Streamlit + Pillow (`src/preprocess.py`) |
| **Studio — mask** | Clothing region + `seg_conf` | PyTorch + Transformers **SegFormer** + SciPy (`src/segmentation.py`) |
| **Studio — gate** | Block try-on if `seg_conf < 0.85` | `src/confidence.py` |
| **Studio — try-on** | Dress the person | **IDM-VTON** Space → **CatVTON** → **SD2** Inference API (`src/tryon.py`) |
| **Studio — try-on score** | `tryon_conf` from seg + CLIP + mask | FashionCLIP + `summarize_scores` |
| **Studio — Top-5** | Similar SKUs | FashionCLIP + `data/embeddings.npy` (`src/recommend.py`) |
| **Studio — copy** | Caption, advice, explain | Gemini API (`src/llm_advisor.py`) |
| **Catalog** | Browse / Save / Buy | Pandas CSV + Streamlit + Google Shopping URLs |
| **Stylist AI** | Avatar JSON, chat, voice, ranked recs | Gemini + FashionCLIP text/image towers |
| **Profile** | Saved pieces | `data/wishlist.json` + catalog rows |

### API keys

| Key | Purpose | Required? |
| --- | --- | --- |
| `HF_TOKEN` | IDM-VTON / CatVTON Spaces + Hub downloads + SD2 fallback | Yes for live try-on |
| `GOOGLE_API_KEY` | Gemini stylist / garment caption / chat / STT | Optional (template fallback) |
| `CONFIDENCE_GATE` | Override the 0.85 threshold (default 0.85) | Optional |

**Note:** No OpenAI key. Shopping uses public Google Shopping search URLs — no Shopping API key.

---

## 7. Models and pretrained weights

VESTURE uses **pretrained checkpoints only**. We run inference / transfer learning. We do **not** train SegFormer, IDM-VTON, FashionCLIP, or Gemini from scratch.

### Which model for which job (with links)

| Job in the app | Field | Model | Links |
| --- | --- | --- | --- |
| Clothing **mask** + `seg_conf` on the person photo | Computer vision (segmentation) | **SegFormer-B2 Clothes** `mattmdjaga/segformer_b2_clothes` | [HF model](https://huggingface.co/mattmdjaga/segformer_b2_clothes) · [paper](https://arxiv.org/abs/2105.15203) · [ATR data](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) |
| Primary **try-on** (person + garment **image**) | Computer vision (diffusion) | **IDM-VTON** Space `yisol/IDM-VTON` | [Space](https://huggingface.co/spaces/yisol/IDM-VTON) · [weights](https://huggingface.co/yisol/IDM-VTON) · [paper](https://arxiv.org/abs/2403.05139) |
| Fallback try-on if the Space is queued | Computer vision (diffusion) | **CatVTON** Space `zhengchong/CatVTON` | [Space](https://huggingface.co/spaces/zhengchong/CatVTON) · [weights](https://huggingface.co/zhengchong/CatVTON) · [paper](https://arxiv.org/abs/2407.15886) |
| Last-resort try-on (**mask + text**) | Computer vision (inpainting) | **SD 2 Inpainting** `stabilityai/stable-diffusion-2-inpainting` | [HF model](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) |
| **Top-5** similar catalog items | Vision–language (embeddings) | **Marqo FashionCLIP** `Marqo/marqo-fashionCLIP` | [HF model](https://huggingface.co/Marqo/marqo-fashionCLIP) · [blog](https://www.marqo.ai/blog/search-model-for-fashion) |
| Backup encoder | Vision–language | **CLIP ViT-B/32** `openai/clip-vit-base-patch32` | [HF model](https://huggingface.co/openai/clip-vit-base-patch32) · [paper](https://arxiv.org/abs/2103.00020) |
| Caption, advice, explain, chat, voice | NLP / multimodal LLM | **Gemini 2.0 Flash** | [API](https://ai.google.dev/) · [generateContent](https://ai.google.dev/api/generate-content) |
| **Buy / Find online** | Search (not a neural net) | **Google Shopping** `tbm=shop` | [Help](https://support.google.com/googleshopping/answer/9128904) |

| Checkpoint / source | Pretrained on (authors) | Runs on |
| --- | --- | --- |
| `mattmdjaga/segformer_b2_clothes` | ATR / 17,706 image–mask pairs, 18 classes | Local PyTorch (CPU or CUDA) |
| `yisol/IDM-VTON` (ECCV 2024) | VITON-HD train; eval VITON-HD + Dress Code | HF Space GPU via `gradio_client` |
| `zhengchong/CatVTON` (ICLR 2025) | ~73k public try-on samples | HF Space GPU |
| `stabilityai/stable-diffusion-2-inpainting` | LAION-style web image–text | HF Inference API |
| `Marqo/marqo-fashionCLIP` (Apache 2.0) | ViT-B-16 LAION-2B + GCL on 1M+ fashion SKUs | Local `open_clip` |
| `openai/clip-vit-base-patch32` | WIT 400M image–text | Local Transformers |
| Google Generative Language API | Closed multimodal pretraining | Cloud API |

Override IDs in `.env` if needed: `VTON_SPACE`, `CATVTON_SPACE`, `HF_INPAINT_MODEL`, `GEMINI_MODEL`.

### 7.1 SegFormer-B2 Clothes (local vision)

- **Architecture:** Mix Transformer **MiT-B2** encoder + lightweight MLP decoder (Xie et al., *SegFormer*, NeurIPS 2021, arXiv:2105.15203). ~27M parameters.
- **Task:** 18-class human / clothing semantic segmentation.
- **How we use it:** `transformers.SegformerForSemanticSegmentation`; upsample logits to the photo; softmax; pick labels by garment region (`upper` → Upper-clothes + Dress; `lower` → Skirt + Pants; `dress` → Dress); dilate + Gaussian feather the binary mask.
- **Confidence:** `seg_conf` = mean softmax on the selected clothing pixels. Hard gate **0.85**.
- **Published metrics** (model card — cite on slides):

| | Acc | IoU |
| --- | ---: | ---: |
| Mean (all 18 classes) | 0.80 | 0.69 |
| Upper-clothes | 0.87 | 0.78 |
| Pants | 0.90 | 0.84 |
| Dress | 0.74 | 0.55 |
| Skirt | 0.76 | 0.65 |
| Eval loss | 0.15 | — |

### 7.2 IDM-VTON (primary try-on)

- **Paper:** Choi et al., *Improving Diffusion Models for Authentic Virtual Try-on in the Wild*, ECCV 2024.
- **Architecture:** **SDXL** inpainting UNet (TryOnNet) + frozen **GarmentNet** (parallel UNet, low-level texture via self-attention) + trainable **IP-Adapter** (high-level garment identity via cross-attention). Detailed Gemini `garment_des` is passed as the text condition.
- **Inputs we send:** person PNG, garment PNG, caption, 30 denoise steps, seed 42, Space API `/tryon`.
- **Why this checkpoint:** garment-**image** conditioning keeps prints and logos; text-only inpaint cannot.
- **License:** CC-BY-NC-SA 4.0 (class demo, not commercial).

### 7.3 CatVTON (try-on fallback)

- **Paper:** Chong et al., *Concatenation Is All You Need for Virtual Try-On with Diffusion Models*, ICLR 2025.
- **Idea:** spatially concatenate person + garment; simplified UNet (~899M total, ~49.6M trainable); no extra ReferenceNet / text encoder at inference.
- **When we call it:** IDM-VTON Space queued or error. Cloth type mapped to `upper` / `lower` / `overall`.

### 7.4 Stable Diffusion 2 Inpainting (last resort)

- Mask-guided fill: **white = replace clothing**, black = keep body/background.
- Prompt from Gemini caption or a template (`src/preprocess.py`). Negative prompt strips extra limbs / watermark artifacts.
- Used only if both Spaces fail, or if no garment image is provided.

### 7.5 Marqo FashionCLIP (recommendation encoder)

- **Base:** OpenCLIP **ViT-B-16** pretrained on LAION-2B (`laion2b_s34b_b88k`).
- **Domain adapt:** Generalized Contrastive Learning (GCL) on titles, descriptions, **color, material, category, style** — not captions alone. ~150M parameters. Trained on **>1M** fashion SKUs.
- **Why not vanilla CLIP:** fashion language (silhouette, knit, kurta, chino) and product-still retrieval. Marqo reports **AvgRecall 0.192 / Recall@1 0.094 / MRR 0.200** vs FashionCLIP 2.0 **0.163 / 0.077 / 0.165** on Atlas, DeepFashion In-shop, DeepFashion Multimodal, Fashion200k, KAGL, Polyvore.
- **Load path:** `open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionCLIP")`. Fallback: Hugging Face CLIP ViT-B/32.

### 7.6 Why this stack

- SegFormer: fashion labels + CPU-friendly.
- IDM-VTON: best open garment-image try-on without a local GPU.
- CatVTON / SD2: queue and API resilience.
- FashionCLIP: SOTA-style fashion retrieval vs generic CLIP.
- Gemini: multimodal stylist layer (image + text + audio) without training an LLM.

### 7.7 How confidence is evaluated

All scores are in **[0, 1]**. Code: `src/segmentation.py`, `src/confidence.py`, `src/recommend.py`. Studio draws them as bars with a **0.85** marker (`CONFIDENCE_GATE`).

These are **proxy scores** (heuristics + cosine), **not** calibrated probabilities like “85% chance the try-on is correct.”

#### Scores we report

| Score | What it measures | How it is evaluated | Gate |
| --- | --- | --- | --- |
| **`seg_conf`** | How sure SegFormer is on the clothing pixels | After softmax over 18 classes, take `max` over the chosen labels (e.g. upper-clothes + dress), then **mean** that probability on pixels predicted as clothing. Empty mask → 0. | **Hard:** if `< 0.85`, try-on is **blocked**; mask is still shown |
| **`mask_quality`** | Whether the mask looks like a single clothing region | `0.5 · coverage_score + 0.5 · connectedness`. Coverage: 1.0 if 5–55% of the image is white; lower if tiny or huge. Connectedness: largest blob / all clothing pixels (penalizes speckles). | Soft: folded into `tryon_conf` (weight 0.2) |
| **`clip_sim`** | Did the try-on keep the garment look? | FashionCLIP encode **garment** (or person mask-crop) and **try-on cropped to the mask bbox**; **cosine** of L2-normalized vectors. If CLIP fails, placeholder 0.5. | Soft: folded into `tryon_conf` (weight 0.4) |
| **`tryon_conf`** | Overall Studio quality | `0.4·seg_conf + 0.4·clip_sim + 0.2·mask_quality`, clipped to [0, 1] | **Soft flag:** `passes_tryon_gate` if ≥ 0.85 (does not block; Gemini is told to be honest if below) |
| **`reco_conf` / `similarity`** | How close a catalog SKU is | Cosine of FashionCLIP query vs `embeddings.npy` row. Color hint can add +0.04–0.08 for ranking only. | Highlight tile if ≥ 0.85 (`high_confidence`) |

```text
seg_conf     = mean( softmax(SegFormer)[chosen_labels].max  on clothing pixels )
mask_quality = 0.5 · coverage_score + 0.5 · (largest_component / clothing_pixels)
clip_sim     = cosine( FashionCLIP(garment) , FashionCLIP(try-on_mask_crop) )
tryon_conf   = 0.4 · seg_conf + 0.4 · clip_sim + 0.2 · mask_quality
reco_conf    = cosine( FashionCLIP(query) , catalog_embedding_i )
```

#### Evaluation order in Studio

```text
1. SegFormer → mask + seg_conf
2. if seg_conf < 0.85  → FAIL pill, stop (no IDM-VTON)
3. IDM-VTON / CatVTON / SD2 → result image
4. clip_sim on mask-crop vs garment
5. mask_quality on the same mask
6. tryon_conf composite → PASS/FAIL vs 0.85 (informational)
7. FashionCLIP Top-5 → each item.similarity (reco_conf)
8. Gemini explain_result reads these numbers (does not invent them)
```

#### Why 0.85

Course requirement: show a confidence threshold. **0.85** is a strict filter on **segmentation** (bad pose / multi-person / tiny subject) so we do not waste a Space call. The same number is reused as a **highlight** on try-on and recommendations so the UI is consistent. It is **not** a published SegFormer accuracy (that Acc/IoU table is from the model card on ATR).

#### What we do **not** evaluate

- No FID / SSIM / LPIPS on VITON-HD (we do not retrain try-on).
- No human A/B study.
- `tryon_conf` is **not** a temperature-scaled softmax of IDM-VTON.

---### 7.8 Fine-tuning (authors vs this project)

**We do not fine-tune.** Authors already fine-tuned the checkpoints we load. We only run **inference** and **prompt adaptation**.

| Checkpoint | What the authors fine-tuned | What we do |
| --- | --- | --- |
| SegFormer-B2 Clothes | Full segmentation train on ATR (17,706 pairs) from MiT-B2 | `from_pretrained` + mask post-process |
| IDM-VTON | SDXL TryOnNet + IP-Adapter on VITON-HD (11,647 pairs) | Call the HF Space |
| CatVTON | ~49.6M / 899M params (self-attention) on ~73k samples | Call the HF Space |
| FashionCLIP | GCL fine-tune of ViT-B-16 LAION-2B on 1M+ SKUs | Encode catalog once → `embeddings.npy` |
| Gemini 2.0 Flash | Google closed pretraining (not public) | System prompts + few-shot + grounding on our scores |

Adaptation techniques that are **not** weight updates: system instructions, few-shot captions, RAG-style grounding on Top-5 titles, morphological mask cleanup, 0.85 gate.

**Viva:** *Authors fine-tuned SegFormer on ATR, IDM-VTON on VITON-HD, FashionCLIP with GCL. We freeze those weights and adapt Gemini with system prompts.*

Future (not built): LoRA on SegFormer, a small contrastive pass on our 95 SKUs, or Vertex Gemini tune if budget allows.

### 7.9 Computer vision vs NLP

| | Computer vision | NLP / LLM | Both (vision–language) |
| --- | --- | --- | --- |
| **Does** | Pixels → mask, try-on image, embeddings | Words / JSON / transcript out | Image and text in one space |
| **Models** | SegFormer, IDM-VTON, CatVTON, SD2 | Gemini (caption, advice, chat, STT) | FashionCLIP, CLIP, Gemini Vision |
| **Classic CV (no net)** | Letterbox, dilate, feather, `mask_quality` | Template SD prompt if no API key | Cosine rank + Shopping URL |

```text
CV:  photo → SegFormer mask → IDM-VTON pixels → FashionCLIP image vector
NLP: Gemini system prompt → caption / advice / chat
VLM: garment JPEG + text → Gemini; query text → FashionCLIP text tower
```

### 7.10 Try-on masks (Studio)

When the user taps **Generate Try-On**:

1. Person: EXIF → RGB → letterbox **768×1024** (fast **384×512**).
2. Garment: letterbox **512×512**.
3. **SegFormer** 18-class map → keep clothing IDs for the chosen region:
   - `upper` → Upper-clothes (4) + Dress (7)
   - `lower` → Skirt (5) + Pants (6)
   - `dress` → Dress (7)
   - If empty: try `[4]`, then `[4,5,6,7]`.
4. Binary mask → **dilate 3 iterations** → **Gaussian σ = 1.5** (soft edge so inpaint does not leave a halo).
5. `seg_conf` = mean softmax on those pixels. If **< 0.85**, Studio shows the mask only and **does not** call try-on.
6. Mask is an `L` image: **white = replace clothing**, **black = keep** face / arms / background.
7. **IDM-VTON / CatVTON** use person + garment images (the Space builds its own internal mask). **SD2** uses our mask as the inpaint region.
8. Studio also shows `colorize_labels` (18-color debug map).
9. `mask_quality` = 0.5·coverage (prefer 5–55% of the frame) + 0.5·largest connected component.
10. `crop_by_mask` crops the try-on to the white bbox so CLIP compares **fabric**, not the wall.

Face, hair, arms, legs stay black in the mask so the diffusion model should not redraw identity.

---

## 8. LLM API details (Google Gemini)

Gemini is used as a **pretrained inference API**. Weights stay on Google’s servers. We adapt it with system prompts, few-shot examples, and grounding on SegFormer / FashionCLIP numbers — not by fine-tuning.

### 8.1 Connection

| Item | Value |
| --- | --- |
| Product | [Google Gemini API](https://ai.google.dev/) (Google AI Studio / Generative Language API) |
| Python SDK | `google-genai` (`from google import genai`) with fallback `google.generativeai` |
| Call | `client.models.generate_content(model=…, contents=…, config=GenerateContentConfig(…))` |
| REST shape | `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` |
| Default model | **`gemini-2.0-flash`** (override with `GEMINI_MODEL`) |
| Auth | `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env` |
| If key missing | Template captions / advice / chat — app still runs |

`gemini-2.0-flash` is natively multimodal: **text + JPEG + audio** in one `contents` list (`types.Part.from_bytes`). No separate speech-to-text vendor.

### 8.2 Calls in `src/llm_advisor.py`

| Function | Modalities | Config | What it returns |
| --- | --- | --- | --- |
| `caption_garment` | garment JPEG + text | temp **0.4**, max tokens **280**, `SYSTEM_STYLIST` | One-sentence `garment_des` for IDM-VTON / SD2 |
| `style_advice` | text (caption, Top-5, scores) | same | Occasion / pairing copy |
| `explain_result` | text (engine, `seg_conf`, CLIP, gate, Top-5) | same | Plain-English result card |
| `analyze_avatar_llm` | person JPEG | temp **0.3**, max tokens **700**, JSON MIME, `SYSTEM_AVATAR` | Color season, undertone, palette, body type |
| `stylist_chat` | text + optional photos + voice | temp **0.6**, max tokens **512**, `SYSTEM_CHAT`, last 8 turns | Conversational stylist |
| `transcribe_audio` | WAV/audio bytes | default generate | Transcript of a voice note |

Grounding rules in the system prompt: do not invent brands; rank catalog titles we actually retrieved; stay inside the avatar palette when it exists.

### 8.3 System prompts (from `src/llm_advisor.py`)

**`SYSTEM_STYLIST`** (caption, advice, explain — temp 0.4, 280 tokens):

```text
You are VESTURE, a world-class luxury fashion stylist with exceptional high-fashion sense
and elite recommendation judgment.
Write 2–4 short sentences. No hashtags. No emoji.
Ground every claim in the garment description, confidence scores,
or recommended catalog titles you are given. Do not invent brands.
```

**`SYSTEM_CHAT`** (Stylist AI — temp 0.6, 512 tokens, last 8 turns): creative-director rules (silhouette, color, fabric, occasion); rank catalog pieces; honor photos and voice notes; stay in the client palette; 2–6 sentences; no invented brands. Includes few-shot: garden wedding / cool undertones, and casual weekend / pear shape.

**`SYSTEM_AVATAR`** (JSON profile — temp 0.3, 700 tokens): analyze the person; kind; no brand names; never comment on attractiveness; **JSON only** (`color_season`, `undertone`, `palette`, `body_type`, …).

**Caption few-shot** sent with the garment JPEG:

```text
Example: a navy cotton crew-neck t-shirt with a relaxed fit and matte finish.
Example: a black tailored blazer in structured wool with notch lapels.
Describe this garment in one sentence for a virtual try-on model.
Include color, fabric, silhouette, and garment type.
```

Full prompt strings: [`MODELS.md`](MODELS.md) §4.

### 8.4 What we did **not** do

- No LoRA / Vertex supervised fine-tune of Gemini.
- No OpenAI, Claude, or local GGUF LLM.
- Caption timeout ~25s; failures fall back to a template such as “a navy cotton crew-neck t-shirt…”.

**Viva:** *Pretrained Gemini via API inference + domain prompting, grounded on our DL outputs — allowed as transfer learning / model inference.*

---

## 9. Datasets

Two layers: **(A)** datasets the **pretrained models** were trained on (citation / slides), and **(B)** data the **app actually reads** at runtime.

### 9.1 Provenance datasets (inside the checkpoints — we do not retrain)

| Dataset | Stats | Paper / source | Which model ate it |
| --- | --- | --- | --- |
| **ATR** (HF `mattmdjaga/human_parsing_dataset`) | **17,706** image–mask pairs; **18** labels (Background, Hat, Hair, Upper-clothes, Skirt, Pants, Dress, …) | Liang et al., Deep Human Parsing | SegFormer-B2 Clothes |
| **VITON-HD** | **13,679** frontal woman + upper-garment pairs, **1024×768**; train **11,647** / test **2,032** | Choi et al., CVPR 2021 | IDM-VTON (and CatVTON) training |
| **Dress Code** | **53,792** garments / **107,584** images (in-shop + worn); pairs ≈ 15,363 upper / 8,951 lower / 29,478 dresses, 1024×768 | Morelli et al., 2022 | Multi-category try-on; IDM-VTON eval; CatVTON |
| **LAION-2B / WIT** | Web-scale image–text | OpenCLIP / CLIP papers | FashionCLIP base + CLIP fallback |
| **Marqo fashion mix (~1M SKUs)** | DeepFashion In-shop **52,591**; DeepFashion Multimodal **42,537**; Fashion200K **201,624**; KAGL **44,434**; Atlas **78,370**; Polyvore **94,096**; iMaterialist **721,065** (eval / GCL mix) | Marqo FashionCLIP card | FashionCLIP domain fine-tune + published retrieval numbers |
| **CUHK DeepFashion** | **800k+** images; 50 categories; 1,000 attributes; 300k+ consumer–shop pairs | Liu et al., CVPR 2016 | Literature benchmark for clothes retrieval (not stored in-repo) |
| **Kaggle Fashion Product Images** | **~44,000** Myntra-style SKUs, titles, colors, article types | Aggarwal / Kaggle | Typical visual-recommendation catalog; same *style* as our DF* titles |

### 9.2 Runtime data the Streamlit app uses

| Asset | Location | Size in this repo | Role |
| --- | --- | --- | --- |
| **Shop catalog** | `data/catalog.csv` + `data/catalog/images/` | **~95 SKUs** (≈ 33 upper / 30 lower / 32 dress): 5 editorial `VE*` looks + `DF*` product stills | FashionCLIP Top-5, Catalog page, Stylist recs, Profile saves |
| **Catalog embeddings** | `data/embeddings.npy` | One L2-normalized vector per CSV row | Cosine retrieval (no re-encode of the whole shop per click) |
| **Studio demo pairs** | `data/samples/person_*.jpg` + `cloth_*.jpg` | Optional VITON-HD-*style* pairs (`python -m src.demo_samples --n 16`) | One-click try-on if the user has no photo |
| **Cached try-on** | `assets/demo/` | Offline PNGs | Presentation fallback if Spaces / HF API are down |
| **Self-collected photos** | User upload | 20–50 recommended for the demo | Live Studio + confidence analysis |

Build / refresh the shop catalog (does **not** train a model):

```bash
python -m src.catalog_builder --from-deepfashion --n 90
```

Download order in `src/catalog_builder.py`:

1. High-res product stills from Hugging Face Dataset Viewer — `PestoRosso/lamoda-fashion-product-images` (min side **512 px**, save JPEG up to 1080×1440).
2. If that is thin, stream `Marqo/deepfashion-inshop` then `Marqo/deepfashion-multimodal`.
3. If the network fails, generate **placeholder** garments (`--placeholders`).

Rows keep `id, title, category, color, image_path, shop_url`. Accessories / footwear / innerwear are skipped so retrieval stays on clothing.

### 9.3 How each dataset is used

| Dataset | Train from scratch in this project? | Loaded at app runtime? | Slides? |
| --- | --- | --- | --- |
| ATR | No (inside SegFormer weights) | Via the checkpoint | Yes — parsing provenance |
| VITON-HD | No | Optional `data/samples/` demos | Yes — try-on benchmark |
| Dress Code | No | No | Yes — multi-category try-on |
| DeepFashion / Lamoda / Kaggle-style products | No | **Yes — subset becomes the catalog** | Yes — recommendation data |
| FashionCLIP 1M mix | No (inside Marqo weights) | Via the checkpoint | Yes — why FashionCLIP beats CLIP |
| Local `data/catalog/` | No | **Yes** | Yes |
| Self-collected | No | **Yes** (uploads) | Yes |

### 9.4 Preprocessing

1. Person: EXIF → RGB → letterbox (768×1024, or 384×512 fast).
2. Mask: SegFormer → class select → dilate → feather.
3. Gate: warn / block if `seg_conf < 0.85`.
4. Garment: RGB → Gemini one-line caption (or template).
5. Catalog: filter apparel → JPEG → FashionCLIP embed → `embeddings.npy`.

---

## 10. Recommendation dataset, deep-learning retrieval, and Google Shopping

Industry fashion recommenders (FashionCLIP paper, *Scientific Reports* 2022; Marqo GCL; Kaggle Fashion Product Images visual-search notebooks) follow the same pattern: **embed catalog images once**, **embed the query**, **rank by cosine similarity**, then **send the shopper to a store**. VESTURE implements that pipeline on a small in-app catalog and opens **Google Shopping** for purchase.

### 10.1 Recommendation dataset (what is ranked)

The retrieval index is **only** `data/catalog.csv` — not the full 800k DeepFashion dump (too large for a class demo). Each row is a shoppable-style product still plus metadata:

```text
id, title, category, color, image_path, shop_url
DF020, Nike Women Purple Polo T-shirt, upper, purple, data/catalog/images/df_upper_020.jpg, https://www.google.com/search?tbm=shop&q=Nike+Women+Purple+Polo+T-shirt+buy
```

- **Visual index:** FashionCLIP image vectors in `embeddings.npy` (row *i* ↔ CSV row *i*), L2-normalized so **dot product = cosine**.
- **Text index:** the same encoder’s text tower (`recommend_from_text`) for Stylist queries such as “garden wedding, cool undertones”.
- **Color boost:** +0.04–0.08 if the row color matches Gemini / avatar palette hints.
- **Output:** Top-**5** dicts with `similarity` (recommendation confidence) and `high_confidence` if cosine ≥ **0.85**.

Queries:

| Query | Encoder input | Typical screen |
| --- | --- | --- |
| After try-on | Garment image (or mask-crop of the result) | Studio “similar items” |
| Avatar styling | Person photo + optional text | Stylist AI |
| Typed / voice request | FashionCLIP **text** embedding | Stylist chat |

This is **content-based visual retrieval** (deep metric learning), not collaborative filtering. No user–item rating matrix.

### 10.2 Deep-learning retrieval math

```text
q ← FashionCLIP.encode(query_image or query_text)     # unit vector
S ← embeddings.npy                                    # N × D, unit rows
score_i = S_i · q   (+ small color boost)
Top-5 = argsort(score) descending
```

Same idea as CLIP product search: one shared space for photos and phrases, so “navy linen shirt” and a navy shirt photo land nearby.

### 10.3 Google Shopping (shop-out layer)

We do **not** scrape Google, call SerpApi, or use the Merchant Center Content API. After FashionCLIP picks a title, **Buy / Find online** is a normal Shopping-tab URL:

```text
https://www.google.com/search?tbm=shop&q={urlencode(title + " buy")}
```

| Piece | Meaning |
| --- | --- |
| `tbm=shop` | Google Search **Shopping** tab (product cards, prices, sellers) |
| `q=` | Product title from the catalog / recommendation |
| Built in | `shop_url_for()` in `src/recommend.py` and `catalog_builder.py` |

Google Shopping ranks live merchant listings by relevance to that query (Google Shopping Help). That is the “find this elsewhere online” step: DL finds **which** garment, Google Shopping finds **where to buy** it.

Flow:

```text
Try-on garment / avatar / text
        │
        ▼
FashionCLIP cosine vs local catalog (DeepFashion-style stills)
        │
        ▼
Top-5 titles + similarity %
        │
        ▼
Gemini explains why they match
        │
        ▼
Buy → Google Shopping search for that title
```

### 10.4 What a larger production system would add

- Full DeepFashion In-shop (52k) or Kaggle 44k index + FAISS / Annoy.
- Optional Google **Vertex AI Search for Retail** or a Shopping API instead of a search URL.
- Click / purchase logs for learning-to-rank (out of scope for BIT4443).

---

## 11. Streamlit Application Features

- Project title: **VESTURE**
- Tabs: **Home** · **Studio** · **Catalog** · **Stylist AI** · **Profile**
- Upload person + garment, or pick a VITON-HD demo pair
- Prediction button: **Generate Try-On**
- Results: before/after, mask, confidence metrics
- Top-5 similar items (FashionCLIP) + **Buy / Find online → Google Shopping**
- LLM panels: AI garment description, stylist advice, result explanation
- Stylist chat: text, image attach, voice (Gemini transcribe)

---

## 12. Project Structure

```text
Virtual_Try_On/
├── app.py                 # Streamlit UI (VESTURE)
├── overview.md            # This document
├── MODELS.md              # Which model / links / fine-tune / prompts / masks
├── README.md
├── requirements.txt
├── .env.example           # HF_TOKEN, GOOGLE_API_KEY
├── src/
│   ├── preprocess.py
│   ├── segmentation.py    # SegFormer
│   ├── tryon.py           # IDM-VTON → CatVTON → SD2
│   ├── confidence.py
│   ├── recommend.py       # FashionCLIP Top-5 + Google Shopping URLs
│   ├── catalog_builder.py # DeepFashion / Lamoda subset + embeddings
│   ├── llm_advisor.py     # Gemini 2.0 Flash caption / advice / chat / STT
│   ├── stylist.py         # Avatar analysis + catalog ranking
│   └── demo_samples.py    # VITON-HD-style Studio pairs
├── data/catalog/
├── data/samples/
├── assets/home/
└── assets/demo/           # Offline try-on fallback images
```

---

## 13. Strengths, Limitations, Future Work

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

- Index the full DeepFashion In-shop (~52k) or Kaggle ~44k set with FAISS
- Optional Vertex Gemini fine-tune on fashion Q&A if budget allows
- Optional Shopping / retail search API instead of `tbm=shop` URLs

---

## 14. Course Alignment (BIT4443)

| Requirement          | How VESTURE meets it                                    |
| -------------------- | ------------------------------------------------------- |
| Real-world problem   | Online try-before-you-buy / returns                     |
| Pretrained DL models | SegFormer-B2, IDM-VTON, CatVTON, SD2, FashionCLIP     |
| Transfer / inference | HF Spaces + Gemini `generateContent`; prompt adaptation |
| Dataset description  | ATR, VITON-HD, Dress Code, DeepFashion/Lamoda catalog   |
| Recommendation data  | Local FashionCLIP index; shop-out via Google Shopping   |
| Streamlit UI         | Title, description, upload, predict, result, confidence |
| Analysis             | Confidence metrics, strengths/limits                    |

---

## 15. Key references (slides / viva)

| Topic | Citation |
| --- | --- |
| SegFormer | Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021. Checkpoint: `mattmdjaga/segformer_b2_clothes` (ATR). |
| ATR parsing | Liang et al., Deep Human Parsing; HF `mattmdjaga/human_parsing_dataset` (17,706 pairs). |
| IDM-VTON | Choi et al., *Improving Diffusion Models for Authentic Virtual Try-on in the Wild*, ECCV 2024, arXiv:2403.05139. Space `yisol/IDM-VTON`. |
| CatVTON | Chong et al., *Concatenation Is All You Need for Virtual Try-On*, ICLR 2025, arXiv:2407.15886. Space `zhengchong/CatVTON`. |
| VITON-HD | Choi et al., CVPR 2021 — 13,679 pairs at 1024×768. |
| Dress Code | Morelli et al., 2022 — 53,792 garments / 107,584 images. |
| DeepFashion | Liu et al., *DeepFashion*, CVPR 2016 — 800k+ images, in-shop + consumer-to-shop retrieval. |
| FashionCLIP (concept) | Chia et al., *Contrastive language and vision learning of general fashion concepts*, Scientific Reports 2022. |
| Marqo FashionCLIP | Fine-tuned ViT-B-16 LAION-2B + GCL; HF `Marqo/marqo-fashionCLIP`. Eval: DeepFashion In-shop (52,591), Multimodal (42,537), Fashion200K, KAGL, Atlas, Polyvore. |
| CLIP fallback | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021. `openai/clip-vit-base-patch32`. |
| Gemini API | Google AI Studio / `google-genai`; `models.generateContent` on `gemini-2.0-flash` (text, JPEG, audio). Docs: https://ai.google.dev/ |
| Google Shopping | Public Search Shopping tab: https://www.google.com/search?tbm=shop&q=… (no merchant API key). |
| Full model map | [`MODELS.md`](MODELS.md) |

---
