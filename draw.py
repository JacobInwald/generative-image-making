from typing import Tuple, List
from PIL import Image, ImageDraw
import numpy as np
import random as r
import cv2
from skimage.metrics import structural_similarity as ssim

class img:

    def __init__(self, path="", from_img=False, w=0,h=0, pxlmap=[], from_map=False):
        self.path = path

        if from_img:
            img_rgba = Image.open(path).convert("RGBA")
        elif from_map:
            img_rgba = Image.fromarray(pxlmap)
        else:
            img_rgba = Image.new('RGBA', (w, h), (255, 255, 255, 255)).convert("RGBA")

        self.width = img_rgba.width
        self.height = img_rgba.height
        self.pixel_map = np.array(img_rgba)
    
    def pix_diff(self, pxlmap, subsample=None):

        if subsample is not None:
            pxlmap = pxlmap[::subsample, ::subsample, :]
            subpxlmap = self.pixel_map[::subsample, ::subsample, :]
            return 1 - ssim(pxlmap, subpxlmap, multichannel=True, channel_axis=2)
        else:
            return 1 - ssim(pxlmap, self.pixel_map, multichannel=True, channel_axis=2)
    
    def save(self):
        Image.fromarray(self.pixel_map).save(self.path)

class canvas:

    def __init__(self, path, width, height):
        self.path = path
        self.shapes = []
        self.w = width
        self.h = height
        self.read()

    def read(self):
        with open(self.path, 'r') as f:
            data = f.readlines()
            for i in range(0, len(data)//6):
                q = quad()
                q.read(self.path, i)
                self.shapes.append(q)
    
    def save(self):
        with open(self.path, 'w') as f:
            f.write('')
        for q in self.shapes:
            q.save(self.path)

    def to_img(self):
        i = img(self.path+'.png', from_img=False, w=self.w, h=self.h)
        for q in self.shapes:
            q.draw(i)
        i.save()
        return i
    


class quad:

    def __init__(self, tl: Tuple[int, int] = (0,0),
                        tr: Tuple[int, int] = (0,0),
                        bl: Tuple[int, int] = (0,0),
                        br: Tuple[int, int] = (0,0),
                        c: Tuple[int,int,int] = (0,0,0),
                        alpha: int = 0):
        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br
        self.c = c
        self.alpha = alpha

    def draw(self, img: img):
        r,g,b = self.c
        a = self.alpha
        colour = (r,g,b,a)
        img_arr = img.pixel_map
        points = [self.tl, self.tr, self.br, self.bl]
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        poly_pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [poly_pts], color=colour)

        overlay = img_arr[mask > 0]
        img_arr[mask > 0] = ((overlay + colour)/2).astype(np.uint8)
        return img_arr
    
    def create_child(self, std: int):
        return quad(self.tl+(r.random()*std, r.random()*std),
                    self.tr+(r.random()*std, r.random()*std),
                    self.bl+(r.random()*std, r.random()*std),
                    self.br+(r.random()*std, r.random()*std),
                    self.c+(r.random()*std, r.random()*std, r.random()*std),
                    self.alpha+r.random()*std)
    
    def combine(self, other):
        child = quad()
        for attr_name in ['tl', 'tr', 'bl', 'br', 'c', 'alpha']:
            if isinstance(getattr(self, attr_name), tuple):
                coeff = r.random()
                attr_value = tuple(
                    coeff * getattr(self, attr_name)[i] + (1 - coeff) * getattr(other, attr_name)[i]
                    for i in range(len(getattr(self, attr_name)))
                )
                setattr(child, attr_name, attr_value)
            else:
                coeff = r.random()
                attr_value = coeff * getattr(self, attr_name) + (1 - coeff) * getattr(other, attr_name)
                setattr(child, attr_name, attr_value)
        return child

    def save(self, file):
        with open(file, "a") as f:
            f.write(str(self.tl)+'\n'+
                    str(self.tr)+'\n'+
                    str(self.bl)+'\n'+
                    str(self.br)+'\n'+
                    str(self.c)+'\n'+
                    str(self.alpha)+'\n')
    
    def read(self, file, index):
        with open(file, "r") as f:
            d = f.readlines()
            d = d[index*6:(index+1)*6]
            self.alpha = int(d[-1].strip()//1)
            d = d[:-1]

            data = []
            for s in d:
                clean = [c for c in s if c not in [' ', '(', ')', '\n']]
                s=""
                for c in clean:
                    s+=c
                data.append(tuple([int(i) for i in s.split(',')]))
            d = data
            self.tl = d[0]
            self.tr = d[1]
            self.bl = d[2]
            self.br = d[3]
            self.c = d[4]

