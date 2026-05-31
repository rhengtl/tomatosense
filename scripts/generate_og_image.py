"""Generate frontend/static/img/og-image.png for social link previews."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent.parent / "frontend" / "static" / "img" / "og-image.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 1200, 630
RED      = (220, 38,  38)   # red-600
RED_DARK = (153, 27,  27)   # red-800
RED_MID  = (185, 28,  28)   # red-700

img  = Image.new("RGB", (W, H), RED)
draw = ImageDraw.Draw(img)

# Decorative circles (bottom-right, top-left)
draw.ellipse([900, 350, 1350, 800],   fill=RED_DARK)
draw.ellipse([-150, -150, 250, 250],  fill=RED_DARK)
draw.ellipse([-80, -80, 160, 160],    fill=RED_MID)

# Horizontal accent band
draw.rectangle([0, H // 2 - 115, W, H // 2 + 115], fill=(185, 28, 28, 200))

try:
    font_title = ImageFont.load_default(size=108)
    font_sub   = ImageFont.load_default(size=46)
    font_tag   = ImageFont.load_default(size=32)
except TypeError:
    font_title = ImageFont.load_default()
    font_sub   = font_title
    font_tag   = font_title

draw.text((W // 2, H // 2 - 52), "TomatoSense",
          fill="white", font=font_title, anchor="mm")
draw.text((W // 2, H // 2 + 44), "AI-Powered Tomato Ripeness Classifier",
          fill=(255, 205, 205), font=font_sub, anchor="mm")
draw.text((W // 2, H // 2 + 96), "Linear SVM  ·  PCA  ·  96.25% Accuracy",
          fill=(255, 170, 170), font=font_tag, anchor="mm")

img.save(OUT, "PNG", optimize=True)
print(f"Generated: {OUT}")
