import imghdr
import dftfilter as fil
from PIL import Image
import numpy as np

img = Image.open('Pic1.jpg')
img = np.array(img)


H = [fil.hp_filter('btw', np.shape(img), i) for i in range(5, 300, 25)]

print(H)

res = [Image.fromarray(fil.filter(img, t)) for t in H]

for ind, img in enumerate(res):
    img = img.convert("L")
    img.save(f".\Res\img{ind}.png")
