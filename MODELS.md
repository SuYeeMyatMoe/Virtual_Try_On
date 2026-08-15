# VESTURE — Models, fine-tuning, prompts, CV / NLP, masks

Companion to [`overview.md`](overview.md). This file is the full map of **which model does which job**, with **links**, **how fine-tuning works**, **system prompts**, **computer vision vs NLP**, and the **Studio try-on mask pipeline**.

Code: `src/segmentation.py`, `src/tryon.py`, `src/preprocess.py`, `src/confidence.py`, `src/recommend.py`, `src/llm_advisor.py`, `app.py`.

---

## 1. Which model for which job (with links)

| Job in the app | Field | Model we call | Checkpoint / API | Official links |
| --- | --- | --- | --- | --- |
| Find clothes on the body + clothing **mask** | **Computer vision** (semantic segmentation) | **SegFormer-B2 Clothes** | `mattmdjaga/segformer_b2_clothes` | [Model card](https://huggingface.co/mattmdjaga/segformer_b2_clothes) · [Paper](https://arxiv.org/abs/2105.15203) · [Train set ATR](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) |
| Primary **virtual try-on** (person + garment **image**) | **Computer vision** (conditional diffusion) | **IDM-VTON** | Space `yisol/IDM-VTON` | [Space demo](https://huggingface.co/spaces/yisol/IDM-VTON) · [Weights](https://huggingface.co/yisol/IDM-VTON) · [Paper](https://arxiv.org/abs/2403.05139) · [GitHub](https://github.com/yisol/IDM-VTON) |
| Fallback try-on if IDM-VTON is queued | **Computer vision** (concatenation diffusion) | **CatVTON** | Space `zhengchong/CatVTON` | [Space demo](https://huggingface.co/spaces/zhengchong/CatVTON) · [Weights](https://huggingface.co/zhengchong/CatVTON) · [Paper](https://arxiv.org/abs/2407.15886) |
| Last-resort try-on using **mask + text** | **Computer vision** (inpainting) | **SD 2 Inpainting** | `stabilityai/stable-diffusion-2-inpainting` | [Model card](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) |
| **Top-5 similar products** (image or text query) | **CV + NLP** (vision–language embeddings) | **Marqo FashionCLIP** | `Marqo/marqo-fashionCLIP` | [Model card](https://huggingface.co/Marqo/marqo-fashionCLIP) · [Blog](https://www.marqo.ai/blog/search-model-for-fashion) · [GitHub](https://github.com/marqo-ai/marqo-FashionCLIP) |
| Backup encoder if FashionCLIP fails | **CV + NLP** | **OpenAI CLIP ViT-B/32** | `openai/clip-vit-base-patch32` | [Model card](https://huggingface.co/openai/clip-vit-base-patch32) · [CLIP paper](https://arxiv.org/abs/2103.00020) |
| Garment caption, stylist copy, explain scores, chat, voice | **NLP** (multimodal LLM) | **Gemini 2.0 Flash** | `gemini-2.0-flash` | [Gemini API](https://ai.google.dev/) · [generateContent](https://ai.google.dev/api/generate-content) · [AI Studio](https://aistudio.google.com/) |
| Open a store listing after retrieval | Search (not a DL model) | **Google Shopping** | `tbm=shop` URL | [How Shopping works](https://support.google.com/googleshopping/answer/9128904) · example `https://www.google.com/search?tbm=shop&q=…` |

### One-line viva map

| You see in Studio / Catalog / Stylist | Model responsible |
| --- | --- |
| Colored **label map** + white clothing **mask** | SegFormer |
| Person wearing the uploaded garment | IDM-VTON (else CatVTON, else SD2) |
| Bars: seg / CLIP / mask quality / try-on conf | SegFormer softmax + FashionCLIP cosine + mask heuristic |
| Top-5 tiles + similarity % | FashionCLIP vs `data/embeddings.npy` |
| “Saved to Catalog” shop **Buy** button | Google Shopping URL from the product title |
| “Navy cotton crew-neck…” caption | Gemini Vision |
| Styling advice + result explanation | Gemini Text |
| Stylist chat / voice note | Gemini Chat + audio transcription |

---

## 2. Computer vision vs NLP in this project

VESTURE is a **hybrid**: vision models perceive pixels; the LLM talks to the shopper. FashionCLIP sits in the middle (one embedding space for images **and** words).

```text
                    COMPUTER VISION                         NLP / LLM
                    ───────────────                         ─────────
Person photo ──► SegFormer ──► mask, labels, seg_conf
                                      │
Garment photo ──► Gemini Vision ────────────────────────► garment caption (text)
                                      │
                 IDM-VTON / CatVTON / SD2 ──► try-on image
                                      │
                 FashionCLIP image tower ──► vector ─┐
                                                     ├─ cosine Top-5
Query text / voice ──► Gemini STT ──► text ──►       │
                 FashionCLIP text tower ──► vector ──┘
                                      │
Scores + titles ──► Gemini Text ──► advice, explanation, chat
                                      │
Titles ──► Google Shopping URL
```

| Layer | What it is | VESTURE modules |
| --- | --- | --- |
| **Computer vision** | Pixels → structure / new pixels | SegFormer, IDM-VTON, CatVTON, SD2, mask dilate/feather, CLIP **image** tower |
| **NLP** | Language in / language out | Gemini system prompts, captions, advice, chat, JSON avatar profile, audio → transcript |
| **Vision–language (VLM)** | Shared image–text space | FashionCLIP / CLIP; Gemini multimodal (JPEG + text + audio) |
| **Classical CV (not a neural net)** | Morphology, scoring | `binary_dilation`, Gaussian feather, `mask_quality`, letterbox resize |

---

## 3. How fine-tuning works (authors vs this project)

**Course wording:** transfer learning + **model inference**. We load **already fine-tuned** checkpoints. We do **not** run a training loop or update weights.

### 3.1 What the original authors fine-tuned

| Checkpoint | Started from | Fine-tuned on | Method |
| --- | --- | --- | --- |
| SegFormer-B2 Clothes | SegFormer MiT-B2 (ImageNet / ADE20K-style pretrain) | [ATR human parsing](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) — 17,706 image–mask pairs, 18 classes | Full semantic-segmentation fine-tune ([training notes](https://github.com/mattmdjaga/segformer_b2_clothes)) |
| IDM-VTON | **SDXL** + IP-Adapter | [VITON-HD](https://github.com/shadow2496/VITON-HD) train **11,647** pairs; eval on VITON-HD + [Dress Code](https://github.com/aimagelab/dress-code) | Train TryOnNet + IP-Adapter; freeze GarmentNet |
| CatVTON | Stable Diffusion UNet / VAE | ~**73k** public try-on samples | Parameter-efficient: **~49.6M / 899M** trainable (self-attention) |
| Marqo FashionCLIP | OpenCLIP **ViT-B-16** on [LAION-2B](https://huggingface.co/Marqo/marqo-fashionCLIP) | **1M+** fashion SKUs (titles, color, material, category) | **Generalized Contrastive Learning (GCL)** |
| CLIP ViT-B/32 | From scratch on WIT 400M | — | Contrastive image–text (Radford et al.) |
| Gemini 2.0 Flash | Google closed pretraining | Not public | We **cannot** download or train these weights |

### 3.2 What VESTURE does instead of fine-tuning

| Technique | Where | Is it fine-tuning? |
| --- | --- | --- |
| Load `from_pretrained(...)` / Space `predict` | SegFormer, FashionCLIP, IDM-VTON | **No** — inference only |
| **System prompts** + few-shot examples | Gemini `GenerateContentConfig.system_instruction` | **No** — prompt adaptation / instruction at inference |
| Ground Gemini on `seg_conf`, CLIP %, catalog titles | `src/llm_advisor.py` | **No** — retrieval-augmented prompting |
| Precompute `embeddings.npy` | Catalog | **No** — index the shop, not the encoder |
| Dilate + feather the mask | `src/segmentation.py` | **No** — classical post-process |
| Confidence gate 0.85 | `src/confidence.py` | **No** — decision rule |

**Viva sentence:** *The authors fine-tuned SegFormer on ATR, IDM-VTON on VITON-HD, and FashionCLIP with GCL. We freeze those weights and adapt Gemini with system prompts plus our CV scores.*

### 3.3 If we *did* fine-tune later (future work, not implemented)

| Model | Realistic student-scale method | Data we would need |
| --- | --- | --- |
| SegFormer | LoRA / decoder-only train on extra poses | Extra labeled masks |
| FashionCLIP | Small contrastive pass on our 95 SKUs | Image–title pairs already in `catalog.csv` |
| Gemini | Vertex AI supervised / LoRA tune | Fashion Q&A pairs (budget) |
| IDM-VTON | Not practical locally (SDXL + GPU weeks) | Full VITON-HD |

---

## 4. System prompts and user prompts (exact text)

All live in `src/llm_advisor.py`. Model: **`gemini-2.0-flash`**. Key: `GOOGLE_API_KEY`. SDK: `google-genai` → `client.models.generate_content`.

### 4.1 `SYSTEM_STYLIST` — caption, advice, explanation

Temperature **0.4**, max tokens **280**.

```text
You are VESTURE, a world-class luxury fashion stylist with exceptional high-fashion sense
and elite recommendation judgment.
Write 2–4 short sentences. No hashtags. No emoji.
Ground every claim in the garment description, confidence scores,
or recommended catalog titles you are given. Do not invent brands.
```

### 4.2 `SYSTEM_CHAT` — Stylist AI conversation

Temperature **0.6**, max tokens **512**, last 8 turns.

```text
You are VESTURE, a world-class personal stylist with exceptional high-fashion sense
and the sharpest outfit-recommendation judgment in luxury retail.
Think like a creative director: silhouette, color theory, fabric, proportion, occasion,
and what will actually flatter this client. Recommend with confidence — what to wear,
why it works, and what to pair. When catalog pieces are listed, rank them and say which
to try first. When the shopper shares a photo, read the garment, color, and fit before advising.
When they send a voice note, treat the transcript as their brief.
If a client profile exists, stay inside that palette and body type.
If nothing has been analyzed yet, still give precise general advice.
Reply in 2–6 short sentences. Do not invent brand names. No hashtags. No emoji.

Example — shopper: I have a garden wedding and cool undertones.
You: Stay in jewel and icy tones — emerald, powder blue, true navy.
A defined-waist midi or a tailored jacket over a fluid skirt will photograph well
and keep the silhouette formal without feeling costume.

Example — shopper: Casual weekend, pear shape.
You: Structure the top — a collared shirt or cropped jacket — and keep trousers
in a continuous dark or mid tone so the hip line stays quiet. Add one leather or
metal accessory so the look does not read flat.
```

### 4.3 `SYSTEM_AVATAR` — color / body JSON

Temperature **0.3**, max tokens **700**, `response_mime_type = application/json`.

```text
You are VESTURE, a world-class personal stylist with exceptional high-fashion sense
and the best recommendation judgment in luxury retail. Analyze the person in the photo.
Be specific and kind. Do not invent brand names. Never comment on attractiveness.
Return JSON only.
```

User prompt asks for keys: `color_season`, `undertone`, `palette`, `avoid_colors`, `color_notes`, `body_type`, `body_notes`, `style_direction`, `silhouette_tips`, `occasions`.

### 4.4 Few-shot garment caption

```text
Example: a navy cotton crew-neck t-shirt with a relaxed fit and matte finish.
Example: a black tailored blazer in structured wool with notch lapels.

Describe this garment in one sentence for a virtual try-on model.
Include color, fabric, silhouette, and garment type. Category hint: {category}.
```

That sentence becomes IDM-VTON `garment_des` / the SD2 prompt.

### 4.5 Other user prompts

| Call | User prompt (summary) |
| --- | --- |
| `style_advice` | Occasion, season, what to pair — given garment text, region, Top-5, `tryon_conf`, `seg_conf` |
| `explain_result` | 2–3 sentences for a shopper — engine, scores, gate, Top-5; honest if below 0.85 |
| `transcribe_audio` | “Transcribe this spoken fashion request exactly. Return only the transcript, no quotes.” |

### 4.6 SD2 prompts (not Gemini) — `src/preprocess.py`

**Positive (template if no Gemini):**  
`photorealistic fashion photo of a person wearing {color} {style} {shirt/pants/dress}, natural lighting, sharp fabric details, realistic wrinkles`

**Negative:**  
`blurry, deformed body, extra limbs, bad anatomy, watermark, text, low quality, cartoon, painting, face distortion`

---

## 5. Try-on pipeline and masks (Studio)

This is what happens when the user taps **Generate Try-On**.

### 5.1 Step-by-step

```text
1. Person photo
      EXIF → RGB → letterbox 768×1024 (or fast 384×512)
2. Garment photo
      EXIF → RGB → letterbox 512×512
3. SegFormer
      18-class label map → pick clothing IDs for upper / lower / dress
      → binary mask → dilate 3× → Gaussian σ=1.5 → L image 0–255
      → seg_conf = mean softmax on those pixels
4. Gate
      if seg_conf < 0.85 → show mask only, do not call try-on
5. Gemini (optional)
      garment JPEG → one-line caption
6. Try-on cascade
      IDM-VTON(person, garment, caption)
        else CatVTON(person, garment, cloth_type)
        else SD2 inpaint(person, MASK, prompt)   ← mask is required here
        else cached assets/demo PNG
7. Score
      CLIP_sim = cosine(FashionCLIP(garment or mask-crop), FashionCLIP(try-on crop))
      mask_quality from coverage + connectedness
      tryon_conf = 0.4·seg + 0.4·CLIP + 0.2·mask_quality
8. Recommend
      FashionCLIP(garment) vs embeddings.npy → Top-5 → Google Shopping URLs
9. Gemini
      advice + explanation grounded on scores and titles
```

### 5.2 What the mask *is*

- **Format:** single-channel (`L`) PIL image, same H×W as the person photo.
- **Convention for SD2:** **white (255) = change this clothing**, **black (0) = keep** face, arms, background.
- **IDM-VTON / CatVTON:** the Space infers its own clothing region; we still build a mask for the **gate**, **Studio preview**, **mask_quality**, and **CLIP crop**.

### 5.3 SegFormer labels (`src/segmentation.py`)

| ID | Name | Used for |
| ---: | --- | --- |
| 0 | Background | kept (not inpainted) |
| 1 | Hat | unused |
| 2 | Hair | kept |
| 3 | Sunglasses | unused |
| 4 | **Upper-clothes** | `upper` mask |
| 5 | **Skirt** | `lower` mask |
| 6 | **Pants** | `lower` mask |
| 7 | **Dress** | `dress` mask (and `upper` fallback) |
| 8 | Belt | unused |
| 9–10 | Shoes | unused |
| 11 | Face | kept |
| 12–15 | Legs / arms | kept |
| 16–17 | Bag / Scarf | unused |

Region → IDs: `upper → [4, 7]`, `lower → [5, 6]`, `dress → [7]`. If nothing lights up: try `[4]`, then `[4, 5, 6, 7]`.

### 5.4 Dilate and feather (why)

Raw SegFormer edges are slightly tight. Inpaint needs a little overlap or you get a halo of the old shirt.

1. Threshold `mask > 0.5`
2. **Binary dilation**, 3 iterations (grow the white region)
3. **Gaussian blur**, σ = 1.5 (soft edge / alpha)
4. Clip to `[0, 1]`, scale to 0–255

### 5.5 Mask quality score

```text
coverage = fraction of pixels that are clothing
cov_score = 1 if coverage in [5%, 55%], else taper to 0
conn_score = size of largest connected component / all clothing pixels
mask_quality = 0.5 · cov_score + 0.5 · conn_score
```

Penalizes empty masks, full-image “everything is clothes”, and speckled junk.

### 5.6 What Studio shows

| Widget | Source |
| --- | --- |
| Clothing mask | `mask` from `segment_clothing` |
| Label map | `colorize_labels(label_map)` — 18-color debug |
| Before / after | original person vs try-on |
| Seg confidence | mean softmax |
| Mask quality | heuristic above |
| CLIP similarity | FashionCLIP on mask-crop vs garment |
| Try-on confidence | weighted sum; gate 0.85 |

### 5.7 CLIP crop by mask

`crop_by_mask` takes the bounding box of white pixels so CLIP compares **fabric**, not the whole studio background.

---

## 6. Recommendation + Google Shopping (short)

1. **Dataset ranked:** `data/catalog.csv` (~95 SKUs from DeepFashion / Lamoda-style product stills). Not the full 800k DeepFashion dump.
2. **Encoder:** FashionCLIP image/text towers → `data/embeddings.npy`.
3. **Rank:** cosine similarity, Top-5, highlight ≥ 0.85.
4. **Buy:** `https://www.google.com/search?tbm=shop&q={title}+buy`

Deep learning chooses **which** item; Google Shopping finds **where to buy** it.

---

## 7. Keys, code, datasets (quick)

| Key | For |
| --- | --- |
| `HF_TOKEN` | Spaces + Hub downloads ([token settings](https://huggingface.co/settings/tokens)) |
| `GOOGLE_API_KEY` | Gemini ([AI Studio](https://aistudio.google.com/)) |

| Dataset | Link | Used by |
| --- | --- | --- |
| ATR | https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset | SegFormer pretrain |
| VITON-HD | https://github.com/shadow2496/VITON-HD | IDM-VTON pretrain; optional Studio demos |
| Dress Code | https://github.com/aimagelab/dress-code | Multi-category try-on literature |
| DeepFashion | https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html | Retrieval literature; catalog style |
| Marqo In-shop | https://huggingface.co/datasets/Marqo/deepfashion-inshop | FashionCLIP eval; catalog builder fallback |
| Lamoda stills | https://huggingface.co/datasets/PestoRosso/lamoda-fashion-product-images | High-res catalog download |

---

## 8. File map

| File | Role |
| --- | --- |
| `src/preprocess.py` | EXIF, letterbox, SD prompts |
| `src/segmentation.py` | SegFormer, labels, dilate/feather, colorize |
| `src/tryon.py` | IDM-VTON → CatVTON → SD2 → demo PNG |
| `src/confidence.py` | `mask_quality`, `tryon_conf`, 0.85 gate |
| `src/recommend.py` | FashionCLIP, cosine Top-5, Shopping URLs |
| `src/llm_advisor.py` | Gemini prompts + API |
| `src/catalog_builder.py` | Catalog images + embeddings |
| `app.py` | Streamlit Studio / Catalog / Stylist |
