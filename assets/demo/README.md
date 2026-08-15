# Demo fallback images

Put cached try-on results here for presentation day, for example:

- `tryon_01.png`
- `tryon_02.png`

If the Hugging Face Inference API is unavailable, `src/tryon.py` loads the first image in this folder as a fallback and shows a warning in Streamlit.

Preferred inputs for live demos (helps confidence ≥ 0.85):

- Frontal full-body photo
- Plain background
- Single person
- Good lighting
