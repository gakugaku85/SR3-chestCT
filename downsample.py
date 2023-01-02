from PIL import Image

# 画像を開く
im = Image.open("../dataset/mask_1794_png_2/1794_mask_png561.png")

# 画像を bicubic でダウンサンプリング
im = im.resize((1100, 1536), resample=Image.BICUBIC)

# 画像を出力
im.save("../dataset/mask_1794_png_2/1794_mask_png561.png")