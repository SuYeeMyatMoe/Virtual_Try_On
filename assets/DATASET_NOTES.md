# Dataset & model notes (for slides)

## Models

### SegFormer-B2 Clothes (`mattmdjaga/segformer_b2_clothes`)
- Architecture: MiT-B2 encoder + MLP decoder
- Task: 18-class human/clothing parsing
- Metrics: mean Acc 0.80, mean IoU 0.69; Upper-clothes Acc 0.87; Pants Acc 0.90
- Adaptation: inference only; category → label IDs → binary mask → dilate/feather
- Confidence: mean softmax over selected clothing pixels (`seg_conf`)

### IDM-VTON (`yisol/IDM-VTON`) — primary try-on
- Architecture: SDXL inpainting + garment encoder / IP-Adapter-style attention
- Adaptation: Hugging Face Space via `gradio_client`; person image + garment image + Gemini `garment_des`
- License: CC-BY-NC-SA 4.0 (class demo)

### CatVTON (`zhengchong/CatVTON`) — fallback
- ICLR 2025 concatenation try-on; Space GPU if IDM-VTON is queued

### Stable Diffusion 2 Inpainting (`stabilityai/stable-diffusion-2-inpainting`)
- Last-resort text-guided inpaint via HF Inference API

### Marqo FashionCLIP (`Marqo/marqo-fashionCLIP`)
- Task: fashion image embeddings / retrieval
- Adaptation: precompute catalog embeddings; cosine Top-5 at runtime
- Fallback: `openai/clip-vit-base-patch32`

### Gemini
- Caption garment, styling advice, explain confidence + Top-5
- Inference + prompting only (no fine-tune)

## Datasets

| Name | Stats | Use in project |
|---|---|---|
| ATR / human_parsing_dataset | 18 parsing classes | SegFormer provenance + mask viz |
| VITON-HD | 13,679 pairs; 1024×768 | Benchmark citation; Studio demo pairs |
| Dress Code | 53,792 garments; 107,584 images | Multi-category justification |
| DeepFashion / fashion-product-images | tens of thousands | Runtime catalog (subset) |
| Local catalog | ~80–150 real product images | Runtime recommendations |
| Self-collected | 20–50 photos recommended | Live demo + confidence analysis |

Do **not** train. Runtime needs: catalog + person/garment photos.

## Preprocessing
1. Person: EXIF → RGB → letterbox resize
2. Mask: SegFormer → class select → dilate → feather
3. Gate: require `seg_conf >= 0.85`
4. Catalog: download subset → FashionCLIP embed → `embeddings.npy`

## Confidence
```
tryon_conf = 0.4 * seg_conf + 0.4 * clip_sim + 0.2 * mask_quality
```
Recommendation confidence = cosine similarity; highlight ≥ 0.85.
