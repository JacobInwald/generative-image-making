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
            # Subsample the pixel maps
            pxlmap = pxlmap[::subsample, ::subsample, :]
            this_pxlmap = self.pixel_map[::subsample, ::subsample, :]
        else:
            this_pxlmap = self.pixel_map
        # Compute the mean squared error
        mse = np.mean((pxlmap - this_pxlmap) ** 2)
        
        # Compute the peak signal-to-noise ratio
        psnr = 10 * np.log10(255 ** 2 / mse)
        return psnr


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
    
    def combine(self, other, noise=10):
        child = quad(self.tl,self.tr,self.bl,self.br, self.c, self.alpha)
        for attr_name in ['tl', 'tr', 'bl', 'br', 'c', 'alpha']:
            if isinstance(getattr(self, attr_name), tuple):
                coeff = r.random()
                attr_value = tuple(
                    int(coeff * getattr(self, attr_name)[i] + (1 - coeff) * getattr(other, attr_name)[i] + noise*r.gauss(0, 1))
                    for i in range(len(getattr(self, attr_name)))
                )
                setattr(child, attr_name, attr_value)
            else:
                coeff = r.random()
                attr_value = int(coeff * getattr(self, attr_name) + (1 - coeff) * getattr(other, attr_name) + noise*r.gauss(0, 1))
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
            self.alpha = int(d[-1].strip())
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
    
    

    def gen_rand_quad(w, h, brush_size):
        tryagain = True
        while tryagain:
            tl = (r.randint(0, w-1), r.randint(0, h-1))
            tr = (r.randint(tl[0], w-1), r.randint(0, h-1))
            top = tr[1] if tr[1] > tl[1] else tl[1]
            bl = (r.randint(0, w-1), r.randint(top, h-1))
            br = (r.randint(bl[0], w-1), r.randint(top, h-1))
            mask = np.zeros((h, w), dtype=np.uint8)
            poly_pts = np.array([tl,tr,bl,br], dtype=np.int32)
            cv2.fillPoly(mask, [poly_pts], color=1)
            tryagain = mask.sum() < brush_size
            
        return quad(tl, tr, bl, br,
                    (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)),
                     r.randint(0,255))
    
    def gen_rand_quad2(w, h, brush_size):
        while True:
            # generate four random points
            points = [(int(r.uniform(0, w-1)), int(r.uniform(0, h-1))) for _ in range(4)]
            
            # calculate the area of the shape defined by the points
            area = 0.5 * abs((points[0][0] * points[1][1] + points[1][0] * points[2][1] +
                            points[2][0] * points[3][1] + points[3][0] * points[0][1]) -
                            (points[1][0] * points[0][1] + points[2][0] * points[1][1] +
                            points[3][0] * points[2][1] + points[0][0] * points[3][1]))
            
            # calculate the aspect ratio of the bounding box
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            bounding_box_aspect_ratio = 1000 if max(y_coords) == min(y_coords) else (max(x_coords) - min(x_coords)) / (max(y_coords) - min(y_coords))
            
            # check if the area and aspect ratio meet the requirements
            if area >= brush_size and \
                min(bounding_box_aspect_ratio,
                     1/bounding_box_aspect_ratio) >= 1/3:
                return quad(points[0],
                            points[1],
                            points[2],
                            points[3],
                            (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)),
                            r.randint(0,255))
            

    def gen_rand_quad3(w, h, brush_size):

        hyp = np.sqrt(brush_size)+r.uniform(-1,1)*3

        tl = (int(r.uniform(hyp,w-1)), int(r.uniform(hyp,h-1)))
        alpha = np.deg2rad(r.uniform(-10,50))
        tr = (int(tl[0]+np.cos(alpha)*hyp), int(tl[1]+np.sin(alpha)*hyp))
        beta = np.deg2rad(r.uniform(-10,50))
        br = (int(tr[0]+np.sin(beta)*hyp), int(tr[1]+np.cos(beta)*hyp))
        gamma = np.deg2rad(r.uniform(0,70))
        bl = (int(br[0]-np.cos(gamma)*hyp), int(br[1]+np.sin(gamma)*hyp))
        q = quad(tl, tr, bl, br,
                    (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)),
                     r.randint(0,255))
        # Image.fromarray(q.draw(img(w=w,h=h))).show()
        return quad(tl, tr, bl, br,
                    (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)),
                     r.randint(0,255))