# enclding: utf-8

import os
import cv2
from PIL import Image

src = "./data" #pokeRGB_black
dst = "./resizedData" # resized

os.mkdir(dst)

for each in os.listdir(src):
    img = cv2.imread(os.path.join(src,each))
    img = cv2.resize(img,(256,256))
    cv2.imwrite(os.path.join(dst,each), img)

src = "./resizedData"
dst = "./resized_black/"

os.mkdir(dst)
for each in os.listdir(src):
    png = Image.open(os.path.join(src,each))
    # print each
    if png.mode == 'RGBA':
        png.load() # required for png.split()
        background = Image.new("RGB", png.size, (0,0,0))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
        background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
    else:
        png.convert('RGB')
        png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
