# Virtual Clothing Try-On

Streamlit app for **BIT4443 Deep Learning**: virtual try-on with pretrained Hugging Face models.

**Pipeline:** SegFormer clothing segmentation → **IDM-VTON** (CatVTON / SD2 fallback) → FashionCLIP Top-5 → optional **Gemini** stylist.

`HF_TOKEN` is required for live try-on (Hugging Face Spaces / Inference API). `GOOGLE_API_KEY` is optional for Gemini captions and advice.

## Features

- Upload full-body person photo + garment, or pick a VITON-HD demo pair
- SegFormer mask with **segmentation confidence**
- Hard gate at **0.85** before try-on
- Garment-**image** try-on via IDM-VTON Space (no local GPU)
- Composite **try-on confidence** = `0.4·seg + 0.4·CLIP + 0.2·mask_quality`
- Top-5 similar items + **Buy / Find online** links
- Gemini garment description, stylist advice, and result explanation (template fallback if no key)

## Models

| Model | HF ID | Runs on |
| --- | --- | --- |
| SegFormer-B2 Clothes | `mattmdjaga/segformer_b2_clothes` | Local CPU |
| IDM-VTON (primary) | `yisol/IDM-VTON` Space | HF Space GPU |
| CatVTON (fallback) | `zhengchong/CatVTON` Space | HF Space GPU |
| Stable Diffusion 2 Inpainting | `stabilityai/stable-diffusion-2-inpainting` | HF Inference API |
| Marqo FashionCLIP | `Marqo/marqo-fashionCLIP` | Local CPU |
| Gemini | Google Generative AI API | Cloud API |

Fallback for recommendations: `openai/clip-vit-base-patch32`.

IDM-VTON / CatVTON weights are **CC-BY-NC-SA 4.0** (fine for a class demo).

## Datasets (referenced / used)

| Dataset | Role |
| --- | --- |
| ATR / `mattmdjaga/human_parsing_dataset` | SegFormer provenance (in the weights) |
| VITON-HD | Try-on training provenance + optional Studio demo pairs |
| Dress Code | Multi-category citation |
| DeepFashion / fashion-product-images | Runtime catalog for Top-5 |
| Self-collected photos | Live demo |

You do **not** train on these. The app only needs a real product **catalog** plus person/garment photos.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # then paste HF_TOKEN and optional GOOGLE_API_KEY
```

Create a free token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and set:

```env
HF_TOKEN=hf_xxx
GOOGLE_API_KEY=   # optional
```

Build a real catalog (DeepFashion-style product photos + FashionCLIP embeddings):

```bash
python -m src.catalog_builder --from-deepfashion --n 90
```

Offline placeholder garments if the download fails:

```bash
python -m src.catalog_builder --placeholders --n 30
```

Optional VITON-HD-style Studio demo pairs:

```bash
python -m src.demo_samples --n 16
```

## Run

```bash
streamlit run app.py
```

## Presentation fallback

If Spaces / the HF Inference API are slow or down, place cached try-on PNGs in `assets/demo/` (e.g. `tryon_01.png`). The app will show a fallback image and a warning.

## Project layout

```text
app.py
src/
  preprocess.py
  segmentation.py
  tryon.py              # IDM-VTON → CatVTON → SD2 → demo
  llm_advisor.py        # Gemini caption / advice / explain
  confidence.py
  recommend.py
  catalog_builder.py
  demo_samples.py
data/catalog/
data/samples/           # optional VITON-HD demo pairs
assets/demo/
```

## Course notes for slides

- Problem: online fashion try-before-you-buy
- Models: pretrained only (inference / transfer learning)
- Confidence gate 0.85 for segmentation; report try-on composite score
- Strengths: garment-conditioned try-on without a local GPU; DL + LLM hybrid
- Limits: Space queues / API latency; pose/occlusion failures; proxy confidence

## Quick Run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # set HF_TOKEN and GOOGLE_API_KEY
python -m src.catalog_builder --from-deepfashion --n 90
python -m src.demo_samples --n 16
streamlit run app.py
```
