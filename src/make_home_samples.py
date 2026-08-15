"""Regenerate Home before/after sample illustrations."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "assets" / "home"


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def person_base(w=420, h=560, shirt=(70, 90, 120), pants=(35, 35, 45), label="Before"):
    img = Image.new("RGB", (w, h), (248, 244, 241))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        r = int(248 - 8 * t)
        g = int(244 - 6 * t)
        b = int(241 - 4 * t)
        d.line([(0, y), (w, y)], fill=(r, g, b))
    cx = w // 2
    d.ellipse((cx - 70, h - 70, cx + 70, h - 40), fill=(230, 224, 220))
    d.ellipse((cx - 38, 70, cx + 38, 145), fill=(220, 185, 160))
    d.pieslice((cx - 40, 55, cx + 40, 115), 180, 360, fill=(45, 35, 30))
    d.rounded_rectangle((cx - 85, 145, cx + 85, 320), radius=28, fill=shirt)
    d.ellipse((cx - 120, 155, cx - 70, 230), fill=shirt)
    d.ellipse((cx + 70, 155, cx + 120, 230), fill=shirt)
    d.ellipse((cx - 125, 200, cx - 85, 255), fill=(220, 185, 160))
    d.ellipse((cx + 85, 200, cx + 125, 255), fill=(220, 185, 160))
    d.rounded_rectangle((cx - 70, 310, cx + 70, 480), radius=18, fill=pants)
    d.rectangle((cx - 8, 330, cx + 8, 480), fill=(248, 244, 241))
    d.ellipse((cx - 75, 470, cx - 15, 505), fill=(25, 25, 28))
    d.ellipse((cx + 15, 470, cx + 75, 505), fill=(25, 25, 28))
    d.text((24, 18), label, fill=(26, 26, 26), font=_font(26))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("sample1_before.png", "sample1_after.png", (70, 90, 120), (200, 55, 70), "Everyday tee", "Coral try-on"),
        ("sample2_before.png", "sample2_after.png", (40, 40, 45), (35, 90, 160), "Black top", "Navy blouse"),
        ("sample3_before.png", "sample3_after.png", (120, 140, 110), (90, 40, 120), "Olive shirt", "Plum blouse"),
    ]
    for bname, aname, before_c, after_c, bl, al in pairs:
        person_base(shirt=before_c, label=bl).save(OUT / bname)
        person_base(shirt=after_c, label=al).save(OUT / aname)
        print(f"Wrote {bname} / {aname}")


if __name__ == "__main__":
    main()
