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
3. Virtually dresses the person with **IDM-VTON** (garment-image conditioned; CatVTON / SD2 / local overlay fallback). **Lower-body** garments skip IDM-VTON (that Space is upper-only) and use CatVTON with `cloth_type=lower`
4. Recommends **Top-5 similar items** with **FashionCLIP**. Stylist AI also filters by **menswear / womenswear**, body type, and color season
5. Uses **Gemini** to caption garments, analyze body tone + silhouette, chat as a stylist, and **transcribe voice** into editable text
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
| 6    | Dress person with the garment image       | IDM-VTON (upper) → CatVTON (lower/dress) → SD2 → local overlay |
| 7    | Score try-on quality                      | Composite confidence    |
| 8    | Retrieve Top-5 similar catalog items      | FashionCLIP + color + department filter |
| 9    | Write stylist advice + result explanation | Gemini Text API         |
| 10   | Stylist AI: body tone, palette, menswear/womenswear | Gemini JSON + CLIP + SegFormer |
| 11   | Chat / voice → editable transcript → recs | Gemini STT + FashionCLIP |
| 12   | Download stylist analysis PDF             | fpdf2 (`src/analysis_report.py`) |
| 13   | Shop similar items                        | **Google Shopping** (`tbm=shop`) Buy / Find online |

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
                    IDM-VTON Space (upper only)
                         CatVTON / SD2 / local overlay
                                      │
                                      ▼
                               Try-on result
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              FashionCLIP      Try-on confidence    Gemini Stylist
              Top-5 + shop      (composite score)   advice + explain
              + department /
                color filter

Avatar photo ──► SegFormer labels + face LAB season
              ──► CLIP man vs woman (presentation)
              ──► Gemini JSON (palette, body_type, presentation)
                      │
                      ▼
              Rank catalog (menswear/womenswear + palette)
                      │
Voice ──► Gemini STT ──► editable chat text ──► stylist reply
```

### 4.2 Deep Learning path (perception)

1. **SegFormer** finds where clothes are on the body (waist-clipped `lower` masks)
2. **IDM-VTON** transfers the garment *image* onto the torso when the region is `upper`; **CatVTON** handles `lower` / `dress` and Space queues; **SD2** then **local overlay** if APIs fail
3. **FashionCLIP** finds visually similar products; Stylist ranking also applies palette and menswear/womenswear filters

### 4.3 LLM path (language / styling)

1. **Gemini Vision** describes the garment → IDM-VTON `garment_des` / SD prompt
2. **Gemini Vision JSON** (`analyze_avatar_llm`) reads skin undertone, 12-season palette, geometric body type, and **presentation** (`man` / `woman` / `unisex`) for shopping department
3. Local fallback (`src/stylist.py`): face LAB season classifier + SegFormer shoulder/waist/hip + FashionCLIP zero-shot man vs woman
4. **Gemini Text** gives occasion pairing and explains Studio scores
5. **Gemini Chat** answers in short markdown (what to wear, why, up to 3 catalog titles). Grounded on palette, body type, and department
6. **Gemini audio** transcribes a voice note; the UI shows the transcript as **editable text** before send
7. **Buy** opens Google Shopping for the ranked SKU

### 4.4 Gemini adaptation (not full fine-tuning)

We use **pretrained Gemini via API inference**, adapted with:

- Fashion-stylist system prompts (friendly shopper copy, JSON avatar schema)
- Few-shot examples
- Multimodal garment JPEG + person JPEG + voice audio
- Grounding on SegFormer / FashionCLIP scores, palette, body type, and `presentation`

**Viva answer:** _We did not fine-tune Gemini weights. We adapted pretrained Gemini through inference + domain prompting, grounded on our deep-learning outputs — allowed under “transfer learning, fine-tuning, or model inference.”_

### 4.5 Technique → file (what is deep learning)

VESTURE is a **hybrid DL project**: pretrained neural nets do perception and retrieval; Gemini is **model inference** (no weight update); some steps are classical CV. We do **not** train from scratch.

| Technique | File | Deep-learning relation |
| --- | --- | --- |
| Preprocess (EXIF, RGB, letterbox) | `src/preprocess.py` | **Not a neural net** — prepares pixels for the models |
| Clothing **mask** + `seg_conf` | `src/segmentation.py` | **Yes — SegFormer** (transformer segmentation). Softmax on 18 classes → binary mask |
| Dilate / feather / `mask_quality` | `src/segmentation.py`, `src/confidence.py` | **Not a neural net** — morphology so the mask is usable |
| 0.85 gate + `tryon_conf` | `src/confidence.py` | **Uses DL outputs** (SegFormer softmax + FashionCLIP cosine) + mask heuristic |
| Virtual try-on | `src/tryon.py` | **Yes — diffusion**: IDM-VTON (SDXL), CatVTON, SD2 inpaint; last resort `run_local_overlay` is a paste, not a net |
| Top-5 + menswear filter | `src/recommend.py` | **Yes — FashionCLIP** (contrastive vision–language). Cosine vs `embeddings.npy` |
| Catalog build + `shop_url` | `src/catalog_builder.py` | FashionCLIP **encodes** SKUs once (inference). Writes Google Shopping URLs |
| Gemini caption / chat / STT / JSON | `src/llm_advisor.py` | **Yes — LLM inference** (pretrained Gemini). Not fine-tuned |
| 12-season + body + recs | `src/stylist.py` | Mix: SegFormer labels + CLIP audience (**DL**) + LAB season (classical) + Gemini JSON |
| Analysis PDF | `src/analysis_report.py` | **Not DL** — fpdf2 export of the analysis |
| UI (Studio / Stylist / Buy) | `app.py` | Calls the files above; **Buy** opens the Shopping URL |

**Course wording:** transfer learning + **model inference**. Authors already fine-tuned SegFormer (ATR), IDM-VTON (VITON-HD), FashionCLIP (GCL). We freeze those weights.

### 4.6 Masking workflow (Studio)

File: `src/segmentation.py` (mask) → `src/confidence.py` (gate) → `src/tryon.py` (use mask).

```text
Person photo
    → EXIF / RGB / letterbox 768×1024          src/preprocess.py
    → SegFormer-B2 (18-class label map)        src/segmentation.py
    → Keep clothing IDs for the region:
         upper → 4 (Upper-clothes) + 7 (Dress)
         lower → 5 (Skirt) + 6 (Pants), waist-clipped
         dress → 7 (Dress)
    → Binary mask → dilate 3× → Gaussian σ=1.5
    → White (255) = change this clothing
      Black (0)   = keep face, arms, background
    → seg_conf = mean softmax on clothing pixels
    → if seg_conf < 0.85 → show mask only; do not call try-on
    → IDM-VTON / CatVTON: Space builds its own region;
         we still send CatVTON layers[0] (our mask or a blank layer)
    → SD2 / local overlay: our mask is the replace region
    → crop_by_mask → FashionCLIP compares fabric, not the wall
```

Studio also shows `colorize_labels` (18-color debug map). Face / hair / arms stay black so try-on should not redraw identity. Full label table: §7.10.

### 4.7 How confidence is calculated

File: `src/confidence.py` (`tryon_confidence`, `mask_quality`). Scores are in **[0, 1]**. They are **proxy scores**, not calibrated probabilities.

| Score | Formula / rule | File |
| --- | --- | --- |
| **`seg_conf`** | Mean of SegFormer **softmax** on pixels predicted as the chosen clothing classes | `src/segmentation.py` |
| **`mask_quality`** | `0.5 · coverage + 0.5 · connectedness`. Coverage = 1 if clothing is **5–55%** of the photo; connectedness = largest blob / all clothing pixels | `src/confidence.py` |
| **`clip_sim`** | Cosine( FashionCLIP(garment) , FashionCLIP(try-on crop by mask) ) | `src/recommend.py` |
| **`tryon_conf`** | **`0.4·seg_conf + 0.4·clip_sim + 0.2·mask_quality`** | `src/confidence.py` |
| **`reco_conf`** | Cosine(query, `embeddings.npy` row). Stylist adds palette **+0.22–0.35**, avoid-color penalty, department **+0.05** | `src/recommend.py` |

**Gates**

- **Hard:** if `seg_conf < 0.85` → no Space call (`evaluate_segmentation_gate`).
- **Soft:** `tryon_conf ≥ 0.85` and Top-5 `similarity ≥ 0.85` only **highlight** (PASS pill / high-confidence tile). They do not block the result.

`CONFIDENCE_GATE` in `.env` defaults to **0.85**. Full bars and order: §7.7.

### 4.8 Dataset — how many

We **do not retrain**. Counts are (A) what the **authors** trained on, and (B) what **this repo** loads.

| Dataset | How many | Used how | Link |
| --- | --- | --- | --- |
| **ATR** human parsing | **17,706** image–mask pairs, **18** classes | Inside SegFormer weights (not stored here) | [HF dataset](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) |
| **VITON-HD** | **13,679** pairs (train **11,647** / test **2,032**), 1024×768 | IDM-VTON pretrain; optional Studio demos | [GitHub](https://github.com/shadow2496/VITON-HD) |
| **Dress Code** | **53,792** garments / **107,584** images | Try-on literature / CatVTON mix | [GitHub](https://github.com/aimagelab/dress-code) |
| **CatVTON train mix** | ~**73,000** public try-on samples | Inside CatVTON weights | [HF model](https://huggingface.co/zhengchong/CatVTON) |
| **Marqo FashionCLIP** | **1M+** fashion SKUs (GCL); eval In-shop **52,591** | Inside FashionCLIP weights | [HF model](https://huggingface.co/Marqo/marqo-fashionCLIP) · [In-shop](https://huggingface.co/datasets/Marqo/deepfashion-inshop) |
| **DeepFashion (full)** | **800k+** images | Citation only — too large to index | [CUHK page](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) |
| **Kaggle Fashion Product Images** | **~44,000** SKUs | Same *style* as our catalog titles | [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) |
| **Lamoda product stills** | High-res shop photos (catalog builder) | Download source for `data/catalog/` | [HF dataset](https://huggingface.co/datasets/PestoRosso/lamoda-fashion-product-images) |
| **Runtime catalog** `data/catalog.csv` | **~95 SKUs** (≈ 33 upper / 30 lower / 32 dress; ~**33** pass menswear filter) | What FashionCLIP actually ranks | In-repo `data/catalog.csv` |
| **Catalog embeddings** | **1 vector per SKU** in `embeddings.npy` | Cosine Top-5 | In-repo `data/embeddings.npy` |
| **Studio demo pairs** | Optional **16** (`python -m src.demo_samples --n 16`) | One-click try-on | Built locally from VITON-HD-style pairs |
| **Self-collected uploads** | **20–50** recommended for the live demo | User photos in Studio / Stylist | — |

Build catalog (does **not** train): `python -m src.catalog_builder --from-deepfashion --n 90`.

**Viva — our recommendation data (not from a pretrained model):** we use a small DeepFashion / Lamoda-style product catalog (~95 items) as our own data. All vision models stay pretrained; we only index that catalog and take user photos. FashionCLIP ranks those SKUs for Studio Top-5, Catalog, and Stylist looks.

| Our rec dataset | Role | Link |
| --- | --- | --- |
| **Lamoda product stills** | High-res photos downloaded into `data/catalog/` | [HF `PestoRosso/lamoda-fashion-product-images`](https://huggingface.co/datasets/PestoRosso/lamoda-fashion-product-images) |
| **Marqo DeepFashion In-shop** | Fallback catalog stills | [HF `Marqo/deepfashion-inshop`](https://huggingface.co/datasets/Marqo/deepfashion-inshop) |
| **Marqo DeepFashion Multimodal** | Fallback catalog stills | [HF `Marqo/deepfashion-multimodal`](https://huggingface.co/datasets/Marqo/deepfashion-multimodal) |
| **DeepFashion (CUHK, full)** | Source family / literature — we do **not** store 800k images | [CUHK project page](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) |
| **Runtime index** | What we actually recommend from | In-repo `data/catalog.csv` (~95 SKUs) + `data/embeddings.npy` |

### 4.9 How Google Shopping is connected

There is **no Google Shopping API key** and **no scrape**. Deep learning only chooses **which** product title. The shop-out is a normal Search URL.

1. FashionCLIP ranks a catalog row (`src/recommend.py`).
2. `shop_url_for(title)` (same file; also `src/catalog_builder.py` when the CSV is built) writes:

```text
https://www.google.com/search?tbm=shop&q={urlencode(title + " buy online")}
```

| Piece | Meaning |
| --- | --- |
| `tbm=shop` | Opens Google **Shopping** tab (cards, prices, sellers) |
| `q=` | The **catalog / recommendation title** (not a merchant product ID) |
| Button | **Buy / Find online** in Studio, Catalog, and Stylist (`app.py`) |

Google then ranks live merchant listings for that query. We do not store prices or seller accounts.

```text
Photo / garment / typed request
        → FashionCLIP (deep learning) → Top-5 titles in our ~95 SKU catalog
        → Buy → browser opens tbm=shop URL
        → Google Shopping shows where to buy a similar item
```

---

## 5. Outcome

| Outcome                   | Description                                                                 |
| ------------------------- | --------------------------------------------------------------------------- |
| **Visual try-on**         | Before/after image of the user “wearing” the garment                        |
| **Confidence scores**     | Segmentation, CLIP similarity, mask quality, try-on composite (≥ 0.85 gate) |
| **Top-5 recommendations** | Studio: FashionCLIP cosine. Stylist: cosine + **color boost**, **avoid-color penalty**, and **menswear/womenswear** filter |
| **Shop actions**          | Each match opens a **Google Shopping** search for that product title        |
| **AI stylist copy**       | Garment description + advice + explanation (with Gemini key)                |
| **Stylist analysis**      | Body tone, recommended color set, silhouette, shopping department (Auto / Woman / Man) |
| **Analysis PDF**          | After Analyze: **Download analysis PDF** (season, body tone, body type, palette, silhouette; no catalog list) |
| **Voice chat**            | Mic → Gemini transcript → edit in the chatbot → send                        |
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
| **huggingface_hub / Inference API** | Download weights; **SD 2 Inpainting** last-resort try-on via `router.huggingface.co/hf-inference` (legacy `api-inference.huggingface.co` is retired) |
| **google-genai** | **Gemini 2.0 Flash**: garment caption, advice, explain, avatar JSON, chat, voice STT |
| **fpdf2** | Stylist **analysis PDF** after Analyze (`src/analysis_report.py`) |
| **python-dotenv** | Load `HF_TOKEN`, `GOOGLE_API_KEY`, `CONFIDENCE_GATE` from `.env` |
| **requests / datasets** | Build catalog from HF Dataset Viewer / DeepFashion-style stills |
| **Google Shopping URL** | Catalog / Top-5 **Buy / Find online** (`tbm=shop`) |

### 6.1 Stack by pipeline stage

| App part | What happens | Stack |
| --- | --- | --- |
| **Studio — upload / preprocess** | Person + garment photos | Streamlit + Pillow (`src/preprocess.py`) |
| **Studio — mask** | Clothing region + `seg_conf` | PyTorch + Transformers **SegFormer** + SciPy (`src/segmentation.py`) |
| **Studio — gate** | Block try-on if `seg_conf < 0.85` | `src/confidence.py` |
| **Studio — try-on** | Dress the person | **IDM-VTON** (upper only) → **CatVTON** `/submit_function` + mask layer → **SD2** router → **local overlay** (`src/tryon.py`) |
| **Studio — try-on score** | `tryon_conf` from seg + CLIP + mask | FashionCLIP + `summarize_scores` |
| **Studio — Top-5** | Similar SKUs | FashionCLIP + `data/embeddings.npy` (`src/recommend.py`) |
| **Studio — copy** | Caption, advice, explain | Gemini API (`src/llm_advisor.py`) |
| **Catalog** | Browse / Save / Buy | Pandas CSV + Streamlit + Google Shopping URLs |
| **Stylist AI** | Body-tone JSON, department, chat, **editable voice**, ranked recs, **analysis PDF** | Gemini + FashionCLIP + SegFormer + fpdf2 (`src/stylist.py`, `src/analysis_report.py`) |
| **Profile** | Saved pieces | `data/wishlist.json` + catalog rows |

### API keys

| Key | Purpose | Required? |
| --- | --- | --- |
| `HF_TOKEN` | IDM-VTON / CatVTON Spaces + Hub downloads + SD2 fallback | Yes for live try-on |
| `GOOGLE_API_KEY` | Gemini stylist / garment caption / chat / **voice STT**. Same key as Google AI Studio “Gemini API key” (`GEMINI_API_KEY` also works) | Optional (template fallback; voice needs the key) |
| `CONFIDENCE_GATE` | Override the 0.85 threshold (default 0.85) | Optional |

**Note:** No OpenAI key. Shopping uses public Google Shopping search URLs — no Shopping API key.

---

## 7. Models and pretrained weights

VESTURE uses **pretrained checkpoints only**. We run inference / transfer learning. We do **not** train SegFormer, IDM-VTON, FashionCLIP, or Gemini from scratch.

### Which model for which job (with links)

| Job in the app | Field | Model | Links |
| --- | --- | --- | --- |
| Clothing **mask** + `seg_conf` on the person photo | Computer vision (segmentation) | **SegFormer-B2 Clothes** `mattmdjaga/segformer_b2_clothes` | [HF model](https://huggingface.co/mattmdjaga/segformer_b2_clothes) · [paper](https://arxiv.org/abs/2105.15203) · [ATR data](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) |
| Primary **try-on** (upper-body garment **image**) | Computer vision (diffusion) | **IDM-VTON** Space `yisol/IDM-VTON` `/tryon` | [Space](https://huggingface.co/spaces/yisol/IDM-VTON) · [weights](https://huggingface.co/yisol/IDM-VTON) · [paper](https://arxiv.org/abs/2403.05139) |
| Fallback try-on (lower/dress + Space queue) | Computer vision (diffusion) | **CatVTON** Space `zhengchong/CatVTON` `/submit_function` | [Space](https://huggingface.co/spaces/zhengchong/CatVTON) · [weights](https://huggingface.co/zhengchong/CatVTON) · [paper](https://arxiv.org/abs/2407.15886) |
| Last-resort try-on (**mask + text**), then overlay | Computer vision (inpainting / paste) | **SD 2 Inpainting** via HF router; `run_local_overlay` | [HF model](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) |
| **Top-5** similar catalog items | Vision–language (embeddings) | **Marqo FashionCLIP** `Marqo/marqo-fashionCLIP` | [HF model](https://huggingface.co/Marqo/marqo-fashionCLIP) · [blog](https://www.marqo.ai/blog/search-model-for-fashion) |
| Menswear vs womenswear from the avatar | Vision–language (zero-shot) | FashionCLIP / CLIP text prompts | Same encoder as Top-5 |
| Backup encoder | Vision–language | **CLIP ViT-B/32** `openai/clip-vit-base-patch32` | [HF model](https://huggingface.co/openai/clip-vit-base-patch32) · [paper](https://arxiv.org/abs/2103.00020) |
| Caption, body-tone JSON, chat, **voice STT** | NLP / multimodal LLM | **Gemini 2.0 Flash** | [API](https://ai.google.dev/) · [generateContent](https://ai.google.dev/api/generate-content) |
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
- **Region limit:** `_idm_supports()` in `src/tryon.py` calls this Space **only for `upper`**. Public IDM-VTON is VITON-HD upper-body; pants/dresses would paint onto the torso.
- **Why this checkpoint:** garment-**image** conditioning keeps prints and logos; text-only inpaint cannot.
- **License:** CC-BY-NC-SA 4.0 (class demo, not commercial).

### 7.3 CatVTON (try-on fallback)

- **Paper:** Chong et al., *Concatenation Is All You Need for Virtual Try-On with Diffusion Models*, ICLR 2025.
- **Idea:** spatially concatenate person + garment; simplified UNet (~899M total, ~49.6M trainable); no extra ReferenceNet / text encoder at inference.
- **When we call it:** IDM-VTON Space queued/error, **or** the garment region is `lower` / `dress` (public IDM-VTON is VITON-HD **upper-body only** — pants would paint onto the shirt).
- **API:** live Space names are `/submit_function` (mask-based), then `/submit_function_flux` only if that endpoint is missing. **`/predict` is not published.** Person input is a Gradio ImageEditor dict; **`layers[0]` must exist** (SegFormer mask, or a blank layer so automasker runs).
- Cloth type mapped to `upper` / `lower` / `overall`. Lower masks are **waist-clipped** in `src/segmentation.py` so trousers do not replace the torso.

### 7.4 Stable Diffusion 2 Inpainting (last resort)

- Mask-guided fill: **white = replace clothing**, black = keep.
- Prompt from Gemini caption or a template (`src/preprocess.py`). Negative prompt strips extra limbs / watermark artifacts.
- HTTP host: **`https://router.huggingface.co/hf-inference/models/{model}`** (`HF_INFERENCE_URL`). The old `api-inference.huggingface.co` host is decommissioned (DNS / 410).
- If Spaces and SD2 both fail and a garment image is present: **`run_local_overlay`** pastes the garment into the mask bbox (spatially correct, not photorealistic) instead of stretching a cached upper-body demo onto pants.

### 7.5 Marqo FashionCLIP (recommendation encoder)

- **Base:** OpenCLIP **ViT-B-16** pretrained on LAION-2B (`laion2b_s34b_b88k`).
- **Domain adapt:** Generalized Contrastive Learning (GCL) on titles, descriptions, **color, material, category, style** — not captions alone. ~150M parameters. Trained on **>1M** fashion SKUs.
- **Why not vanilla CLIP:** fashion language (silhouette, knit, kurta, chino) and product-still retrieval. Marqo reports **AvgRecall 0.192 / Recall@1 0.094 / MRR 0.200** vs FashionCLIP 2.0 **0.163 / 0.077 / 0.165** on Atlas, DeepFashion In-shop, DeepFashion Multimodal, Fashion200k, KAGL, Polyvore.
- **Load path:** `open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionCLIP")`. Fallback: Hugging Face CLIP ViT-B/32.
- **Stylist ranking extras** (`src/recommend.py`): color synonym boost (`color_weight` 0.22 avatar / 0.35 chat), **avoid_colors** penalty, **audience** filter from title/category (`Men` / `Women` / dress-like → woman; kidswear dropped for adult recs). Zero-shot `infer_clothing_audience` compares the avatar to “a photograph of a man” vs “a photograph of a woman”.

### 7.6 Why this stack

- SegFormer: fashion labels + CPU-friendly.
- IDM-VTON: best open garment-image try-on without a local GPU (upper-body).
- CatVTON / SD2 / overlay: lower-body, queues, and dead Inference-API host.
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
| **`reco_conf` / `similarity`** | How close a catalog SKU is | Cosine of FashionCLIP query vs `embeddings.npy` row. Palette hints add **+0.22–0.35**; avoid-colors subtract the same weight. Matching menswear/womenswear titles get **+0.05**; wrong department and kidswear are **dropped**. | Highlight tile if ≥ 0.85 (`high_confidence`) |

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
3. IDM-VTON (upper) / CatVTON (lower + fallback) / SD2 / local overlay → result image
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

---

### 7.8 Fine-tuning (authors vs this project)

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
| `analyze_avatar_llm` | person JPEG | temp **0.3**, max tokens **700**, JSON MIME, `SYSTEM_AVATAR` | 12-season, undertone, palette, avoid_colors, body_type, **presentation** |
| `stylist_chat` | text + optional photos | temp **0.6**, max tokens **512**, `SYSTEM_CHAT`, last 8 turns | Markdown shopper reply (≤90 words, up to 3 catalog lines) |
| `transcribe_audio` | WAV / WebM / OGG / MP4 bytes | temp **0.0**, max tokens **400**; MIME sniff + model retry | `(transcript, error)` — UI stages text for edit before send |
| `local_stylist_reply` | (no API) | — | Template markdown if Gemini is down |

Grounding: do not invent brands; rank catalog titles we actually retrieved; stay inside the avatar palette; **menswear never lists a women's dress / kurta / skirt / blouse**. Voice needs `GOOGLE_API_KEY` (same key as Google AI Studio “Gemini API key”; `GEMINI_API_KEY` also works). Caption / chat still fall back to templates without a key.

### 8.3 System prompts (from `src/llm_advisor.py`)

**`SYSTEM_STYLIST`** (caption, advice, explain — temp 0.4, 280 tokens):

```text
You are VESTURE, a world-class luxury fashion stylist with exceptional high-fashion sense
and elite recommendation judgment.
Write 2–4 short sentences. No hashtags. No emoji.
Ground every claim in the garment description, confidence scores,
or recommended catalog titles you are given. Do not invent brands.
```

**`SYSTEM_CHAT`** (Stylist AI — temp 0.6, 512 tokens, last 8 turns): friendly personal shopper, markdown, under 90 words; numbered list of up to 3 catalog titles; menswear vs womenswear rules. Few-shots: Soft Summer colors, and a fashion-show look for a man. Full string: [`MODELS.md`](MODELS.md) §4.2.

**`SYSTEM_AVATAR`** (JSON profile — temp 0.3, 700 tokens): analyze the person; kind; no brand names; never comment on attractiveness; set **`presentation`** to `man` / `woman` / `unisex` for shopping department; **do not default to Light Spring**. JSON keys: `color_season`, `undertone`, `palette`, `avoid_colors`, `color_notes`, `body_type`, `presentation`, `body_notes`, `style_direction`, `silhouette_tips`, `occasions`.

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
- Voice STT has no template fallback — it needs `GOOGLE_API_KEY`.

**Viva:** *Pretrained Gemini via API inference + domain prompting, grounded on our DL outputs — allowed as transfer learning / model inference.*

### 8.5 Stylist AI analysis, voice, and recs (`src/stylist.py`)

**Analyze (body tone → color set → looks)**

1. SegFormer label map: face (id 11) median RGB for undertone; arms only if no face; hair for contrast.
2. Local **12-season** classifier (`_season_from_rgb`): LAB hue / value / chroma → season, undertone, palette, `avoid_colors`. Does not default to Light Spring.
3. Geometric **body type** from shoulder / waist / hip bands on the label map.
4. **`presentation`:** FashionCLIP zero-shot man vs woman (`infer_clothing_audience`); dress/skirt vs pants prior; Gemini JSON `presentation`; UI **Shop for** = Auto / Woman / Man (`resolve_presentation` — override wins).
5. Gemini JSON merge (`analyze_avatar`): if Gemini says Light Spring but the local season differs, keep the **local** season/palette.
6. Body copy is collapsed to **≤3 sentences** (`compact_body_copy`).
7. `catalog_for_avatar` ranks the shop with palette weight **0.22**, avoid-color penalty, and department filter.

**Voice**

Mic audio is **not** sent straight into chat. `transcribe_audio` sniffs WAV / WebM / OGG / MP4, retries Gemini models and MIME types, rejects clips under **2500 bytes**. The transcript lands in an **editable** text area (Send / Discard). `app.py` reloads `src.llm_advisor` before STT so Streamlit cache does not keep a stale function.

**Chat recs**

`catalog_for_query` enriches the typed (or transcribed) request with “menswear men's clothing…” or “womenswear…”, palette, season, body type; FashionCLIP text tower; color weight **0.35**; same audience filter. Chat lists **up to 3** titles.

**Analysis PDF**

After Analyze, **Download analysis PDF** (`src/analysis_report.py`, fpdf2) writes a one-page report: title, season + undertone, shop department, avatar, **01 Body tone** (notes + body type), recommended color set / ease-off swatches, **02 Silhouette**. Catalog picks are **not** included. Chat stays empty until the shopper types.

---

## 9. Datasets

Two layers: **(A)** datasets the **pretrained models** were trained on (citation / slides), and **(B)** data the **app actually reads** at runtime.

### 9.1 Provenance datasets (inside the checkpoints — we do not retrain)

| Dataset | Stats | Paper / source | Which model ate it | Link |
| --- | --- | --- | --- | --- |
| **ATR** | **17,706** image–mask pairs; **18** labels (Background, Hat, Hair, Upper-clothes, Skirt, Pants, Dress, …) | Liang et al., Deep Human Parsing | SegFormer-B2 Clothes | [HF `mattmdjaga/human_parsing_dataset`](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) |
| **VITON-HD** | **13,679** frontal woman + upper-garment pairs, **1024×768**; train **11,647** / test **2,032** | Choi et al., CVPR 2021 | IDM-VTON (and CatVTON) training | [GitHub `shadow2496/VITON-HD`](https://github.com/shadow2496/VITON-HD) |
| **Dress Code** | **53,792** garments / **107,584** images (in-shop + worn); pairs ≈ 15,363 upper / 8,951 lower / 29,478 dresses, 1024×768 | Morelli et al., 2022 | Multi-category try-on; IDM-VTON eval; CatVTON | [GitHub `aimagelab/dress-code`](https://github.com/aimagelab/dress-code) |
| **LAION-2B / WIT** | Web-scale image–text | OpenCLIP / CLIP papers | FashionCLIP base + CLIP fallback | [LAION](https://laion.ai/blog/laion-5b/) · [CLIP / WIT](https://github.com/openai/CLIP) |
| **Marqo fashion mix (~1M SKUs)** | DeepFashion In-shop **52,591**; DeepFashion Multimodal **42,537**; Fashion200K **201,624**; KAGL **44,434**; Atlas **78,370**; Polyvore **94,096**; iMaterialist **721,065** (eval / GCL mix) | Marqo FashionCLIP card | FashionCLIP domain fine-tune + published retrieval numbers | [Model](https://huggingface.co/Marqo/marqo-fashionCLIP) · [In-shop](https://huggingface.co/datasets/Marqo/deepfashion-inshop) · [Multimodal](https://huggingface.co/datasets/Marqo/deepfashion-multimodal) |
| **CUHK DeepFashion** | **800k+** images; 50 categories; 1,000 attributes; 300k+ consumer–shop pairs | Liu et al., CVPR 2016 | Literature benchmark for clothes retrieval (not stored in-repo) | [Project page](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) |
| **Kaggle Fashion Product Images** | **~44,000** Myntra-style SKUs, titles, colors, article types | Aggarwal / Kaggle | Typical visual-recommendation catalog; same *style* as our DF* titles | [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) |
| **Lamoda fashion stills** | High-res product photos (min side 512 px in our builder) | HF Dataset Viewer | Runtime catalog download (`src/catalog_builder.py`) | [HF `PestoRosso/lamoda-fashion-product-images`](https://huggingface.co/datasets/PestoRosso/lamoda-fashion-product-images) |

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

**Viva:** this ~95-item subset is the **recommendation dataset**. We did not train FashionCLIP on it — we only embed and rank it.

### 9.3 How each dataset is used

| Dataset | Train from scratch in this project? | Loaded at app runtime? | Slides? |
| --- | --- | --- | --- |
| ATR | No (inside SegFormer weights) | Via the checkpoint | Yes — parsing provenance |
| VITON-HD | No | Optional `data/samples/` demos | Yes — try-on benchmark |
| Dress Code | No | No | Yes — multi-category try-on |
| DeepFashion / Lamoda / Kaggle-style products | No | **Yes — subset becomes the catalog** | Yes — recommendation data |
| FashionCLIP 1M mix | No (inside Marqo weights) | Via the checkpoint | Yes — why FashionCLIP beats CLIP |
| Local `data/catalog/` (~95 SKUs) | No | **Yes — recommendation index** | Yes |
| Self-collected | No | **Yes** (uploads) | Yes |

### 9.4 Preprocessing

1. Person: EXIF → RGB → letterbox (768×1024, or 384×512 fast).
2. Mask: SegFormer → class select → dilate → feather.
3. Gate: warn / block if `seg_conf < 0.85`.
4. Garment: RGB → Gemini one-line caption (or template).
5. Catalog: filter apparel → JPEG → FashionCLIP embed → `embeddings.npy`.

### 9.5 Dataset links (slides / viva)

| Dataset | URL |
| --- | --- |
| ATR (SegFormer train set) | https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset |
| VITON-HD | https://github.com/shadow2496/VITON-HD |
| Dress Code | https://github.com/aimagelab/dress-code |
| DeepFashion (CUHK) | https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html |
| Marqo DeepFashion In-shop | https://huggingface.co/datasets/Marqo/deepfashion-inshop |
| Marqo DeepFashion Multimodal | https://huggingface.co/datasets/Marqo/deepfashion-multimodal |
| Lamoda product stills (our rec catalog download) | https://huggingface.co/datasets/PestoRosso/lamoda-fashion-product-images |
| Kaggle Fashion Product Images | https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset |
| **Our recommendation catalog** (~95 SKUs in-repo) | `data/catalog.csv` — built from Lamoda + DeepFashion In-shop / Multimodal |
| Marqo FashionCLIP (weights + card) | https://huggingface.co/Marqo/marqo-fashionCLIP |
| SegFormer-B2 Clothes (weights) | https://huggingface.co/mattmdjaga/segformer_b2_clothes |
| IDM-VTON Space / weights | https://huggingface.co/spaces/yisol/IDM-VTON · https://huggingface.co/yisol/IDM-VTON |
| CatVTON Space / weights | https://huggingface.co/spaces/zhengchong/CatVTON · https://huggingface.co/zhengchong/CatVTON |

---

## 10. Recommendation dataset, deep-learning retrieval, and Google Shopping

Industry fashion recommenders (FashionCLIP paper, *Scientific Reports* 2022; Marqo GCL; Kaggle Fashion Product Images visual-search notebooks) follow the same pattern: **embed catalog images once**, **embed the query**, **rank by cosine similarity**, then **send the shopper to a store**. VESTURE implements that pipeline on a small in-app catalog and opens **Google Shopping** for purchase.

### 10.1 Recommendation dataset (what is ranked)

**Viva:** we use a small DeepFashion / Lamoda-style product catalog (~95 items) as our own data. All vision models stay pretrained; we only index that catalog and take user photos. That index is what FashionCLIP recommends from (Studio Top-5, Catalog, Stylist). Sources: [Lamoda](https://huggingface.co/datasets/PestoRosso/lamoda-fashion-product-images) · [DeepFashion In-shop](https://huggingface.co/datasets/Marqo/deepfashion-inshop) · [DeepFashion Multimodal](https://huggingface.co/datasets/Marqo/deepfashion-multimodal) · [CUHK DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).

The retrieval index is **only** `data/catalog.csv` — not the full 800k DeepFashion dump (too large for a class demo). Each row is a shoppable-style product still plus metadata:

```text
id, title, category, color, image_path, shop_url
DF020, Nike Women Purple Polo T-shirt, upper, purple, data/catalog/images/df_upper_020.jpg, https://www.google.com/search?tbm=shop&q=Nike+Women+Purple+Polo+T-shirt+buy
```

- **Visual index:** FashionCLIP image vectors in `embeddings.npy` (row *i* ↔ CSV row *i*), L2-normalized so **dot product = cosine**.
- **Text index:** the same encoder’s text tower (`recommend_from_text`) for Stylist queries.
- **Studio Top-5:** cosine vs the garment image (`recommend_top_k`, default color weight 0.08 if hints are passed; Studio currently passes none).
- **Stylist ranking** (`src/recommend.py` `_rank_catalog`):
  - Palette synonym **boost** (`color_weight` **0.22** avatar / **0.35** chat).
  - **`avoid_colors`** subtract the same weight.
  - Matching menswear/womenswear titles **+0.05**.
  - **Drop** the other department and **kidswear** (`_audience_ok`). Dress / skirt / kurta titles count as womenswear.
  - Catalog **~95** SKUs; about **33** pass the man filter (no dresses).
- **Output:** Top-**5** dicts with `similarity` and `high_confidence` if cosine ≥ **0.85**. Chat surfaces up to **3** titles.

Queries:

| Query | Encoder input | Filters | Typical screen |
| --- | --- | --- | --- |
| After try-on | Garment image (or mask-crop) | Cosine only | Studio “similar items” |
| Analyze avatar | Person photo | Palette + avoid + `presentation` | Stylist looks board |
| Typed / voice request | FashionCLIP **text** (query + department words) | Palette + avoid + audience | Stylist chat |

This is **content-based visual retrieval** (deep metric learning), not collaborative filtering. No user–item rating matrix.

### 10.2 Deep-learning retrieval math

```text
q ← FashionCLIP.encode(query_image or query_text)     # unit vector
S ← embeddings.npy                                    # N × D, unit rows
score_i = S_i · q
        + color_weight · palette_match
        − color_weight · avoid_match
        + 0.05 · department_match
drop if wrong department or kidswear
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
FashionCLIP cosine vs local catalog
        + palette / avoid / menswear filter (Stylist)
        │
        ▼
Top-5 titles + similarity %
        │
        ▼
Gemini explains why they match (chat: up to 3 titles)
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
- Stylist **Analyze:** body tone, recommended color set, silhouette, **Shop for** Auto / Woman / Man
- Stylist looks ranked by FashionCLIP + palette + menswear/womenswear
- **Download analysis PDF** after Analyze (season, body tone, body type, palette, silhouette; no catalog list)
- Stylist chat: stays empty until you type; markdown replies; catalog titles as a numbered list (max 3)
- Voice: Gemini STT → **editable** chat text → Send / Discard (does not auto-send)

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
│   ├── tryon.py           # IDM-VTON (upper) → CatVTON → SD2 router → overlay
│   ├── confidence.py
│   ├── recommend.py       # FashionCLIP Top-5 + audience + color + Shopping URLs
│   ├── catalog_builder.py # DeepFashion / Lamoda subset + embeddings
│   ├── llm_advisor.py     # Gemini caption / JSON avatar / chat / STT
│   ├── stylist.py         # 12-season, presentation, catalog_for_*
│   ├── analysis_report.py # Stylist analysis PDF (fpdf2)
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
- DL + LLM hybrid: try-on, body-tone analysis, department-aware recs, editable voice
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
| Recommendation data  | ~95-SKU DeepFashion/Lamoda catalog (`data/catalog.csv`); FashionCLIP index + palette / menswear filters; shop-out via Google Shopping |
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
| DeepFashion | Liu et al., *DeepFashion*, CVPR 2016 — 800k+ images, in-shop + consumer-to-shop retrieval. [CUHK](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html). |
| **Our rec catalog** | ~95 SKUs in `data/catalog.csv` (not a pretrained train set). Download: [Lamoda](https://huggingface.co/datasets/PestoRosso/lamoda-fashion-product-images), [In-shop](https://huggingface.co/datasets/Marqo/deepfashion-inshop), [Multimodal](https://huggingface.co/datasets/Marqo/deepfashion-multimodal). FashionCLIP only **indexes** this for Top-5 / Stylist. |
| FashionCLIP (concept) | Chia et al., *Contrastive language and vision learning of general fashion concepts*, Scientific Reports 2022. |
| Marqo FashionCLIP | Fine-tuned ViT-B-16 LAION-2B + GCL; HF `Marqo/marqo-fashionCLIP`. Eval: DeepFashion In-shop (52,591), Multimodal (42,537), Fashion200K, KAGL, Atlas, Polyvore. |
| CLIP fallback | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021. `openai/clip-vit-base-patch32`. |
| Gemini API | Google AI Studio / `google-genai`; `models.generateContent` on `gemini-2.0-flash` (text, JPEG, audio). Docs: https://ai.google.dev/ |
| Google Shopping | Public Search Shopping tab: https://www.google.com/search?tbm=shop&q=… (no merchant API key). |
| Full model map | [`MODELS.md`](MODELS.md) |

---
