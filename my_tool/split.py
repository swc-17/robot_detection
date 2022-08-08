import os
import random

def save_txt(ls, fp):
    ls.sort()
    with open(fp, 'w') as f:
        for img in ls:
            f.write(img[:-4]+'\n')


imgs = os.listdir('img') 
labels = os.listdir('label') 
print(len(imgs))
print(len(labels))
num = len(imgs)
split = int(0.8 * num)
random.shuffle(imgs)
save_txt(imgs[:split], 'train.txt')
save_txt(imgs[split:], 'val.txt')
