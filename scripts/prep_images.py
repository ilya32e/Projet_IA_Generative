# -*- coding: utf-8 -*-
"""Rogne les marges uniformes des captures d'écran et écrit des versions *_t.png."""
import glob
import os
from PIL import Image, ImageChops

SRC = os.path.join(os.path.dirname(__file__), "..", "assets", "screenshots")
SRC = os.path.abspath(SRC)


def trim(im, tol=14, pad=8):
    im = im.convert("RGB")
    bg = Image.new("RGB", im.size, im.getpixel((1, 1)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -tol)
    bbox = diff.getbbox()
    if not bbox:
        return im
    l, t, r, b = bbox
    l = max(0, l - pad); t = max(0, t - pad)
    r = min(im.width, r + pad); b = min(im.height, b + pad)
    return im.crop((l, t, r, b))


def main():
    for f in glob.glob(os.path.join(SRC, "*.png")):
        if f.endswith("_t.png"):
            continue
        im = Image.open(f)
        out = trim(im)
        dst = f[:-4] + "_t.png"
        out.save(dst)
        print(f"{os.path.basename(dst)}: {im.size} -> {out.size}")


if __name__ == "__main__":
    main()
