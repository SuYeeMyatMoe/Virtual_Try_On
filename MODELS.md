# VESTURE — Models, fine-tuning, prompts, CV / NLP, masks

Companion to [`overview.md`](overview.md). This file is the full map of **which model does which job**, with **links**, **how fine-tuning works**, **system prompts**, **computer vision vs NLP**, and the **Studio try-on mask pipeline**.

Code: `src/segmentation.py`, `src/tryon.py`, `src/preprocess.py`, `src/confidence.py`, `src/recommend.py`, `src/llm_advisor.py`, `src/stylist.py`, `app.py`.

---

## 1. Which model for which job (with links)

| Job in the app | Field | Model we call | Checkpoint / API | Official links |
| --- | --- | --- | --- | --- |
| Find clothes on the body + clothing **mask** | **Computer vision** (semantic segmentation) | **SegFormer-B2 Clothes** | `mattmdjaga/segformer_b2_clothes` | [Model card](https://huggingface.co/mattmdjaga/segformer_b2_clothes) · [Paper](https://arxiv.org/abs/2105.15203) · [Train set ATR](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) |
| Primary **virtual try-on** (upper-body garment **image**) | **Computer vision** (conditional diffusion) | **IDM-VTON** | Space `yisol/IDM-VTON` `/tryon` | [Space demo](https://huggingface.co/spaces/yisol/IDM-VTON) · [Weights](https://huggingface.co/yisol/IDM-VTON) · [Paper](https://arxiv.org/abs/2403.05139) · [GitHub](https://github.com/yisol/IDM-VTON) |
| Fallback try-on (lower/dress + Space queue) | **Computer vision** (concatenation diffusion) | **CatVTON** | Space `zhengchong/CatVTON` `/submit_function` | [Space demo](https://huggingface.co/spaces/zhengchong/CatVTON) · [Weights](https://huggingface.co/zhengchong/CatVTON) · [Paper](https://arxiv.org/abs/2407.15886) |
| Last-resort try-on using **mask + text**, then paste | **Computer vision** (inpainting / overlay) | **SD 2 Inpainting** + `run_local_overlay` | `stabilityai/stable-diffusion-2-inpainting` via `router.huggingface.co/hf-inference` | [Model card](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) |
| **Top-5 similar products** (image or text query) | **CV + NLP** (vision–language embeddings) | **Marqo FashionCLIP** | `Marqo/marqo-fashionCLIP` | [Model card](https://huggingface.co/Marqo/marqo-fashionCLIP) · [Blog](https://www.marqo.ai/blog/search-model-for-fashion) · [GitHub](https://github.com/marqo-ai/marqo-FashionCLIP) |
| Menswear vs womenswear from the avatar | **CV + NLP** (zero-shot CLIP) | FashionCLIP / CLIP | same encoder | Prompts: “a photograph of a man” vs “a photograph of a woman” |
| Backup encoder if FashionCLIP fails | **CV + NLP** | **OpenAI CLIP ViT-B/32** | `openai/clip-vit-base-patch32` | [Model card](https://huggingface.co/openai/clip-vit-base-patch32) · [CLIP paper](https://arxiv.org/abs/2103.00020) |
| Caption, body-tone JSON, chat, **voice STT** | **NLP** (multimodal LLM) | **Gemini 2.0 Flash** | `gemini-2.0-flash` | [Gemini API](https://ai.google.dev/) · [generateContent](https://ai.google.dev/api/generate-content) · [AI Studio](https://aistudio.google.com/) |
| Open a store listing after retrieval | Search (not a DL model) | **Google Shopping** | `tbm=shop` URL | [How Shopping works](https://support.google.com/googleshopping/answer/9128904) · example `https://www.google.com/search?tbm=shop&q=…` |

### One-line viva map

| You see in Studio / Catalog / Stylist | Model responsible |
| --- | --- |
| Colored **label map** + white clothing **mask** | SegFormer |
| Person wearing the uploaded garment | IDM-VTON if **upper**; else CatVTON; else SD2; else local overlay |
| Bars: seg / CLIP / mask quality / try-on conf | SegFormer softmax + FashionCLIP cosine + mask heuristic |
| Top-5 tiles + similarity % | FashionCLIP vs `data/embeddings.npy` (+ Stylist palette / department) |
| “Saved to Catalog” shop **Buy** button | Google Shopping URL from the product title |
| Body tone + recommended color set | Local 12-season LAB + Gemini JSON (`analyze_avatar`) |
| Shop for Auto / Woman / Man | CLIP audience + Gemini `presentation` + UI override |
| “Navy cotton crew-neck…” caption | Gemini Vision |
| Styling advice + result explanation | Gemini Text |
| Stylist chat | Gemini Chat (`SYSTEM_CHAT`) or `local_stylist_reply` |
| Voice note | Gemini STT → **editable** chat text (not auto-send) |

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
                 IDM-VTON (upper) / CatVTON / SD2 / overlay ──► try-on image
                                      │
                 FashionCLIP image tower ──► vector ─┐
                                                     ├─ cosine Top-5
                                                     │  + palette / avoid / audience (Stylist)
Query text / voice ──► Gemini STT ──► editable text ─► │
                 FashionCLIP text tower ──► vector ──┘
                                      │
Avatar ──► face LAB season + CLIP man/woman + Gemini JSON
                                      │
Scores + titles ──► Gemini Text ──► advice, explanation, chat
                                      │
Titles ──► Google Shopping URL
```

| Layer | What it is | VESTURE modules |
| --- | --- | --- |
| **Computer vision** | Pixels → structure / new pixels | SegFormer, IDM-VTON, CatVTON, SD2, mask dilate/feather, CLIP **image** tower |
| **NLP** | Language in / language out | Gemini system prompts, captions, advice, chat, JSON avatar (`presentation`), audio → transcript |
| **Vision–language (VLM)** | Shared image–text space | FashionCLIP / CLIP (retrieval + man vs woman); Gemini multimodal (JPEG + text + audio) |
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
| Load `from_pretrained(...)` / Space APIs | SegFormer, FashionCLIP, IDM-VTON `/tryon`, CatVTON `/submit_function` | **No** — inference only |
| **System prompts** + few-shot examples | Gemini `GenerateContentConfig.system_instruction` | **No** — prompt adaptation / instruction at inference |
| Ground Gemini on `seg_conf`, CLIP %, catalog titles, palette, `presentation` | `src/llm_advisor.py`, `src/stylist.py` | **No** — retrieval-augmented prompting |
| Precompute `embeddings.npy` | Catalog | **No** — index the shop, not the encoder |
| 12-season LAB + menswear title filter | `src/stylist.py`, `src/recommend.py` | **No** — classical / heuristic ranking |
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
You are VESTURE, a friendly personal shopper. Talk like a helpful person, not a runway critic.
Use everyday words: shirt, trousers, jacket, dress — not 'column', 'editorial', 'atelier', or 'silhouette' unless needed.
Reply in markdown with short lines:
1) One sentence: what to wear, in plain language.
2) One sentence: why it suits their colors or body.
3) If catalog pieces are listed, a numbered list of up to 3, each on its own line:
   '1. **Title** — short reason'. Do not dump every title into one sentence.
Keep the whole reply under 90 words. Blank line between the plan and the list.
If presentation is man/menswear, only men's or unisex pieces — never a women's dress, kurta, skirt, or blouse.
If presentation is woman/womenswear, skip men's-only pieces.
Do not open with 'Menswear,' or the season name alone.
Never tell them to upload an avatar or tap Analyze if a profile is already present.
No hashtags. No emoji.

Example — shopper: What colors suit me? Profile: Soft Summer, dusty rose, sage, taupe, navy.
You: Your coloring is a cool Soft Summer.

**Good on you:** dusty rose, sage, taupe, and soft navy.

**Skip:** hot pink, orange, and stark white.
Example — shopper: I want to wear for a fashion show. Profile: inverted triangle, Soft Summer, man.
You: For a show, wear an easy open shirt with straight navy trousers — nothing boxy on the shoulders.

Those Soft Summer colors stay quiet on your skin.

1. **John Players Men Navy Blue Shirt** — open neck, easy shoulder
2. **Peter England Men Party Blue Jeans** — clean straight line
```

If Gemini is down, `local_stylist_reply` uses the same profile (palette, body, `presentation`) and lists up to 3 catalog titles.

### 4.3 `SYSTEM_AVATAR` — color / body JSON

Temperature **0.3**, max tokens **700**, `response_mime_type = application/json`.

```text
You are VESTURE, a world-class personal stylist with exceptional high-fashion sense
and the best recommendation judgment in luxury retail. Analyze the person in the photo.
Be specific and kind. Do not invent brand names. Never comment on attractiveness.
Set presentation to man or woman from the person in the photo so the catalog can shop
menswear vs womenswear. Silhouette labels stay geometric.
Do not default to Light Spring — that season is overused.
Return JSON only.
```

User prompt keys: `color_season` (one of 12 seasons), `undertone` (warm|cool|neutral), `palette` (5 names), `avoid_colors` (3 names), `color_notes`, `body_type` (geometric), **`presentation` (man|woman|unisex)**, `body_notes`, `style_direction`, `silhouette_tips`, `occasions`.

Local merge in `src/stylist.py`: if Gemini returns Light Spring but the LAB season differs, keep the **photo** season. Body copy is capped at 3 sentences. `presentation` can be overridden in the UI (**Shop for**).

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
| `transcribe_audio` | “Transcribe the spoken words in this audio exactly. Return only the transcript, no quotes, no extra commentary.” Returns `(transcript, error)`. MIME sniff (WAV/WebM/OGG/MP4/MPEG); retry Flash models. Rejects clips under 2500 bytes. Needs `GOOGLE_API_KEY`. |

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
6. Try-on cascade (`src/tryon.py`)
      if region is upper and caption is not pants-like:
          IDM-VTON /tryon (person, garment, caption)
      CatVTON /submit_function (ImageEditor + layers[0] mask, cloth_type upper|lower|overall)
        else /submit_function_flux if that API exists
        (/predict is not published on the live Space)
      else SD2 inpaint via router.huggingface.co/hf-inference (person, MASK, prompt)
      else run_local_overlay (paste garment into mask bbox)
      else cached assets/demo PNG
7. Score
      CLIP_sim = cosine(FashionCLIP(garment or mask-crop), FashionCLIP(try-on crop))
      mask_quality from coverage + connectedness
      tryon_conf = 0.4·seg + 0.4·CLIP + 0.2·mask_quality
8. Recommend
      Studio: FashionCLIP(garment) vs embeddings.npy → Top-5 → Google Shopping URLs
      Stylist: same cosine + palette boost + avoid penalty + menswear/womenswear filter
9. Gemini
      advice + explanation grounded on scores and titles
```

### 5.2 What the mask *is*

- **Format:** single-channel (`L`) PIL image, same H×W as the person photo.
- **Convention for SD2:** **white (255) = change this clothing**, **black (0) = keep** face, arms, background.
- **IDM-VTON / CatVTON:** the Space infers its own clothing region; we still build a mask for the **gate**, **Studio preview**, **mask_quality**, **CLIP crop**, and CatVTON **`layers[0]`**. Lower masks are waist-clipped so trousers do not replace the torso.

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

## 6. Recommendation + Google Shopping

1. **Dataset ranked:** `data/catalog.csv` (~95 SKUs from DeepFashion / Lamoda-style product stills). Not the full 800k DeepFashion dump. About **33** titles pass the menswear filter.
2. **Encoder:** FashionCLIP image/text towers → `data/embeddings.npy`.
3. **Studio rank:** cosine similarity, Top-5, highlight ≥ 0.85.
4. **Stylist rank** (`catalog_for_avatar` / `catalog_for_query` in `src/stylist.py`):
   - `catalog_audience(title, category)` from `Men` / `Women` keywords; `dress` / `skirt` / `kurta` → woman.
   - `_audience_ok`: man → man + unisex (drop dresses and kidswear); woman → woman + unisex (drop men's-only).
   - Palette boost **0.22** (avatar) / **0.35** (chat); `avoid_colors` penalty; department match **+0.05**.
   - Text queries are enriched with “menswear men's clothing shirt trousers” or “womenswear women's clothing”.
   - Chat shows **up to 3** titles; the board keeps Top-5.
5. **Buy:** `https://www.google.com/search?tbm=shop&q={title}+buy+online`

Deep learning chooses **which** item; Google Shopping finds **where to buy** it. Department is clothing catalog, not an identity lecture.

## 6.1 Stylist analysis + voice (short)

| Step | What runs |
| --- | --- |
| Body tone | Face median RGB → LAB 12-season (`src/stylist.py`) |
| Gemini JSON | `analyze_avatar_llm` + Light Spring override if local season differs |
| Silhouette | SegFormer shoulder / waist / hip → geometric body type |
| Department | CLIP man vs woman + dress/skirt prior + **Shop for** override |
| Voice | Gemini STT with MIME retry → editable `st.text_area` → Send / Discard |

`GOOGLE_API_KEY` from [AI Studio](https://aistudio.google.com/apikey) **is** the Gemini key (`GEMINI_API_KEY` also works). Caption/chat have template fallbacks; **voice does not**.

---

## 7. Keys, code, datasets (quick)

| Key | For |
| --- | --- |
| `HF_TOKEN` | Spaces + Hub downloads ([token settings](https://huggingface.co/settings/tokens)) |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Gemini caption / JSON / chat / **STT** ([AI Studio](https://aistudio.google.com/apikey) — same key) |

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
| `src/segmentation.py` | SegFormer, labels, dilate/feather, waist-clipped lower masks |
| `src/tryon.py` | IDM-VTON (upper) → CatVTON `/submit_function` → SD2 router → overlay |
| `src/confidence.py` | `mask_quality`, `tryon_conf`, 0.85 gate |
| `src/recommend.py` | FashionCLIP, cosine Top-5, `catalog_audience`, CLIP man/woman, Shopping URLs |
| `src/llm_advisor.py` | Gemini prompts, JSON avatar, chat, `transcribe_audio` |
| `src/stylist.py` | 12-season LAB, `presentation`, `catalog_for_avatar` / `catalog_for_query` |
| `src/catalog_builder.py` | Catalog images + embeddings |
| `app.py` | Streamlit Studio / Catalog / Stylist (editable voice draft, Shop for) |
