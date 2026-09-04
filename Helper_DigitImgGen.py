from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("test_digits", exist_ok=True)

for digit in range(10):
    img = Image.new("L", (28, 28), 0)  # noir
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((8, 3), str(digit), fill=255, font=font)  # blanc
    img.save(f"test_digits/{digit}.png")

print("Done - 10 images dans test_digits/")