"""Create a simple demo try-on fallback image for presentation day."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "assets" / "demo" / "tryon_01.png"


def main():
    w, h = 384, 512
    img = Image.new("RGB", (w, h), (245, 248, 252))
    draw = ImageDraw.Draw(img)

    # Background gradient-like bands
    for y in range(h):
        c = 230 + int(15 * y / h)
        draw.line([(0, y), (w, y)], fill=(c, c + 5, c + 10))

    # Person silhouette
    cx = w // 2
    draw.ellipse((cx - 36, 50, cx + 36, 120), fill=(220, 190, 170))  # head
    draw.rounded_rectangle((cx - 70, 120, cx + 70, 300), radius=24, fill=(40, 90, 160))  # torso (new shirt)
    draw.rectangle((cx - 55, 300, cx + 55, 460), fill=(40, 40, 50))  # pants
    draw.ellipse((cx - 80, 140, cx - 40, 220), fill=(220, 190, 170))  # arms
    draw.ellipse((cx + 40, 140, cx + 80, 220), fill=(220, 190, 170))

    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small = font

    draw.text((24, 20), "Demo try-on fallback", fill=(30, 40, 55), font=font)
    draw.text((24, h - 36), "Replace with a real cached result for class", fill=(70, 80, 95), font=small)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
