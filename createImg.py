import numpy as np
from PIL import Image

# img = np.array(Image.open('../lena.jpg'))
img = []

# img_gray = img[0:16,0:16,1]
for i in range(64):
    if i%4 == 0:
        img = np.append(img, [0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,
        0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,
        0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,
        0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255])
    else:
        img = np.append(img, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# img.dtype = np.uint8
img = img.reshape(64, 64)
print(img.shape)
img_gray = np.array(img, dtype=np.uint8)
print(img_gray)

np.savetxt('../img.txt', img_gray, fmt='%.1f')



# img_gray = np.array([[0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255]for i in range(16)], dtype=np.uint8)
# print(img_gray)
pil_img_gray = Image.fromarray(img_gray)
pil_img_gray.mode = 'L'
# print(pil_img_gray)
gaku = pil_img_gray.resize(size=[16, 16], resample=Image.BICUBIC)
# print(gaku)

gaku_image = np.array(gaku)
print(gaku_image)
np.savetxt('../gaku.txt', gaku_image, fmt='%.1f')

pil_img_gray.save('../lene16.jpg')
gaku.save('../gaku4.jpg')