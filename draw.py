from typing import Tuple
from PIL import Image
import numpy as np
import random as r
import cv2
from enum import Enum
import scipy
# PIX DIFF TYPES
class IMAGE_DIFF_METHOD(Enum):
    psnr    = 1
    mse     = 2
    ncc     = 3
    euc     = 4
    man     = 5


def random_rgb():
    # hue = r.uniform(0.0, 1.0)
    # saturation = r.uniform(0.4, 1.0)
    # value = r.uniform(0.4, 1.0)
    # a = r.randint(1,255)

    # rd, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    # rd = max(int(rd * 255), 1)
    # g = max(int(g * 255), 1)
    # b = max(int(b * 255), 1)
    rd = int(r.uniform(1, 255))
    g = int(r.uniform(1, 255))
    b = int(r.uniform(1, 255))
    a = int(r.uniform(1, 255))
    return (rd, g, b, a)



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
    
    def pix_diff(self, pxlmap, diff_type=IMAGE_DIFF_METHOD.psnr, subsample=None):
        pxlmap = pxlmap[:, :, :3]
        this_pxlmap = self.pixel_map[:, :, :3]
        if subsample is not None:
            # Subsample the pixel maps
            pxlmap = pxlmap[::subsample, ::subsample, :]
            this_pxlmap = this_pxlmap[::subsample, ::subsample, :]
        i1 = pxlmap 
        i2 = this_pxlmap

        mse = np.mean((i1 - i2) ** 2)

        # Compute the mean squared error
        if diff_type == IMAGE_DIFF_METHOD.mse:
            return mse
        # Compute the peak signal-to-noise ratio 
        elif diff_type == IMAGE_DIFF_METHOD.psnr:
            return 10 * np.log10(255 ** 2 / mse)
        # Compute the normalized cross-correlation
        elif diff_type == IMAGE_DIFF_METHOD.ncc:
            return np.sum( (i1 - np.mean(i1)) * (i2 - np.mean(i2)) ) / ((i1.size - 1) * np.std(i1) * np.std(i2) )
        # Compute the euclidean distance
        elif diff_type == IMAGE_DIFF_METHOD.euc:
            return np.sqrt(np.sum((i1 - i2)^2)) / i1.size
        # Compute the manhattan distance
        elif diff_type == IMAGE_DIFF_METHOD.man:
            return np.sum(abs(i1 - i2)) / i1.size

        return mse
    

    def save(self):
         Image.fromarray(self.pixel_map).save(self.path)


    def show(self):
        Image.fromarray(self.pixel_map).show()

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
                q = quad_point()
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

    def __init__(self,  pos: Tuple[int,int] = (0,0),
                        alpha: float = 0,
                        beta: float = 0,
                        gamma: float = 0,
                        hyp: int = 1,
                        c: Tuple[int,int,int,int] = (1,1,1,1)):
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hyp = hyp
        self.tl = pos
        self.tr = (int(self.tl[0]+np.cos(alpha)*hyp), int(self.tl[1]+np.sin(alpha)*hyp))
        self.br = (int(self.tr[0]+np.sin(beta)*hyp), int(self.tr[1]+np.cos(beta)*hyp))
        self.bl = (int(self.br[0]-np.cos(gamma)*hyp), int(self.br[1]+np.sin(gamma)*hyp))


    def draw(self, img: img):
        img_arr = img.pixel_map
        points = [self.tl, self.tr, self.br, self.bl]
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        poly_pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [poly_pts], color=self.c)

        overlay = img_arr[mask > 0]
        img_arr[mask > 0] = ((overlay + self.c)/2).astype(np.uint8)
        return img_arr
    
    def combine(self, other, noise=10):
        coeff = r.gauss(0.5,0.5)
        tl      = tuple( max(int(coeff * self.tl[i] + (1 - coeff) * other.tl[i] + noise*r.gauss(0, 1)), 0)
                            for i in range(len(self.tl)))
        alpha   = max(min(coeff * self.alpha + (1 - coeff) * other.alpha + noise*r.gauss(-1, 1), 0.5), -0.5)
        beta    = max(min(coeff * self.beta + (1 - coeff) * other.beta + noise*r.gauss(-1, 1), 0.5), -0.5)
        gamma   = max(min(coeff * self.gamma + (1 - coeff) * other.gamma + noise*r.gauss(-1, 1), 0.5), -0.5)
        hyp     = max(int(coeff * self.hyp + (1 - coeff) * other.hyp + noise*r.gauss(0, 1)), 5)
        c = tuple(min(255, max(int(coeff * self.c[i] + (1 - coeff) * other.c[i] + noise*r.gauss(0, 1)), 1))
                    for i in range(len(self.c)))

        return quad(tl,alpha,beta,gamma,hyp,c)

    def save(self, file):
        with open(file, "a") as f:
            f.write(str(self.tl)+'\n'+
                    str(self.c)+'\n'+
                    str(self.alpha)+'\n'+
                    str(self.beta)+'\n'+
                    str(self.gamma)+'\n'+
                    str(self.hyp)+'\n')
    
    def read(self, file, index):
        with open(file, "r") as f:
            d = f.readlines()
            d = d[index*6:(index+1)*6]
            self.hyp = int(d[-1])
            self.gamma = float(d[-2])
            self.beta = float(d[-3])
            self.alpha = float(d[-4])
            d = d[:-4]

            data = []
            for s in d:
                clean = [c for c in s if c not in [' ', '(', ')', '\n']]
                s=""
                for c in clean:
                    s+=c
                data.append(tuple([int(i) for i in s.split(',')]))
            d = data
            self.tl = d[0]
            self.c = d[1]
            new = quad(self.tl, self.alpha,self.beta,self.gamma,self.hyp,self.c)
            self.tr = new.tr
            self.bl = new.bl
            self.br = new.br

    
    def move_inbounds(self, w, h):
        tl, tr, bl, br = self.tl, self.tr, self.bl, self.br

        # Find the minimum and maximum x-coordinates and y-coordinates of the four points
        x_coords = [tl[0], tr[0], bl[0], br[0]]
        y_coords = [tl[1], tr[1], bl[1], br[1]]
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        if not(min_x>=0 and min_y>=0 and max_x <=w and max_y <=h):
            # Calculate the width and height of the object
            obj_w = max_x - min_x
            obj_h = max_y - min_y

            # Calculate the x-coordinate and y-coordinate adjustments
            x_adj = -min(min_x, obj_w)
            y_adj = -min(min_y, obj_h)

            # Apply the adjustments to all four points
            self.tl = (tl[0] + x_adj, tl[1] + y_adj)
            self.tr = (tr[0] + x_adj, tr[1] + y_adj)
            self.bl = (bl[0] + x_adj, bl[1] + y_adj)
            self.br = (br[0] + x_adj, br[1] + y_adj)

    def gen_rand_quad(w, h, brush_size):

        hyp = np.sqrt(brush_size)
        hyp += r.random()*(min(w,h)-hyp)
        hyp = int(hyp)
        hyp = min(hyp, min(w,h))

        tl = (int(r.uniform(0,w-hyp)), int(r.uniform(0,h-hyp)))
        alpha = r.uniform(-0.5,0.5)
        beta = r.uniform(-0.5,0.5)
        gamma = r.uniform(-0.5,0.5)

        new = quad(tl, alpha, beta, gamma, hyp, random_rgb())
        # new.tr = (int(r.uniform(new.tl[0],w)), int(r.uniform(0,h)))
        # new.bl = (int(r.uniform(0,w)), int(r.uniform(max(new.tl[1],new.tr[1]),h)))
        # new.br = (int(r.uniform(new.tl[0],w)), int(r.uniform(max(new.tl[1],new.tr[1]),h)))
        new.move_inbounds(w,h)
        return new


'''
quad class but defined with points instead of angles and lengths. 
The points are defined in clockwise order starting from the top left point
'''
class quad_point:
    
    def __init__(self, ps:list[Tuple[int,int]]=[(1,-1),(1,1),(-1,1),(-1, -1)], c:Tuple[int,int,int,int]=(255,255,255,255)):
        self.points = ps
        self.c = c
        self.tl = self.points[0]
        self.tr = self.points[1]
        self.br = self.points[2]
        self.bl = self.points[3]

    def combine(self, other, noise=0):
        coeff = r.gauss(0.5,0.5)
        points = [tuple(max(int(coeff * self.points[x][i] + (1 - coeff) * other.points[x][i] + noise*r.gauss(0, 1)), 0)
                            for i in range(len(self.points[x])))
                            for x in range(len(self.points))]
        c = tuple(min(255, max(int(coeff * self.c[i] + (1 - coeff) * other.c[i] + noise*r.gauss(0, 1)), 1))
                    for i in range(len(self.c)))
        return quad_point(points, c)


    def gen_rand_quad(w, h, brush_size):
        hyp = np.sqrt(brush_size)
        hyp += r.random()*(min(w,h)-hyp)
        hyp = int(hyp)
        hyp = min(hyp, min(w,h))

        tl = (int(r.uniform(0,w-hyp)), int(r.uniform(0,h-hyp)))
        points = [tl, (tl[0]+hyp, tl[1]), (tl[0]+hyp, tl[1]+hyp), (tl[0], tl[1]+hyp)]

        return quad_point(points, random_rgb())



    # Utility Functions   
    #  
    def draw(self, img: img):
        img_arr = img.pixel_map
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        poly_pts = np.array(self.points, dtype=np.int32)
        cv2.fillPoly(mask, [poly_pts], color=self.c)

        overlay = img_arr[mask > 0]
        img_arr[mask > 0] = ((overlay + self.c)/2).astype(np.uint8)
        return img_arr
    

    def save(self, file):
        with open(file, "a") as f:
            f.write(str(self.points)+'\n'+
                    str(self.c)+'\n')
    

    def read(self, file, index):
        with open(file, "r") as f:
            d = f.readlines()
            d = d[index*2:(index+1)*2]
            for s in d[0][1:-2].split('), '):
                if s == '':
                    continue
                s = s.replace('(', '')
                s = s.replace(')', '')
                self.points.append(tuple([float(i) for i in s.split(',')]))
            self.c = tuple([int(i) for i in d[1].replace('(', '').replace(')', '').split(', ')])


'''
quad class but defined with points instead of angles and lengths. 
The points are defined in clockwise order starting from the top left point
'''
class quad_point_no_alpha:
    
    def __init__(self, ps:list[Tuple[int,int]]=[(1,-1),(1,1),(-1,1),(-1, -1)], c:Tuple[int,int,int,int]=(255,255,255,255)):
        self.points = ps
        self.c = c
        self.tl = self.points[0]
        self.tr = self.points[1]
        self.br = self.points[2]
        self.bl = self.points[3]

    def combine(self, other, noise=0):
        coeff = r.uniform(0.25,0.75)
        points = [tuple(max(int(coeff * self.points[x][i] + (1 - coeff) * other.points[x][i] + noise*r.gauss(0, 1)), 0)
                            for i in range(len(self.points[x])))
                            for x in range(len(self.points))]
        c = tuple(min(255, max(int(coeff * self.c[i] + (1 - coeff) * other.c[i] + noise*r.gauss(0, 1)), 1))
                    for i in range(len(self.c)))
        
        return quad_point(points, c)


    def gen_rand_quad(w, h, brush_size):
        hyp = np.sqrt(brush_size)
        hyp += r.random()*(min(w,h)-hyp)
        hyp = int(hyp)
        hyp = min(hyp, min(w,h))

        tl = (int(r.uniform(0,w-hyp)), int(r.uniform(0,h-hyp)))
        points = [tl, (tl[0]+hyp, tl[1]), (tl[0]+hyp, tl[1]+hyp), (tl[0], tl[1]+hyp)]
        return quad_point(points, random_rgb())



    # Utility Functions   
    #  
    def draw(self, img: img):
        img_arr = img.pixel_map
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        poly_pts = np.array(self.points, dtype=np.int32)
        (rd,g,b,_) = self.c
        c = (rd,g,b,255)
        cv2.fillPoly(mask, [poly_pts], color=c)

        overlay = img_arr[mask > 0]
        img_arr[mask > 0] = ((overlay + c)/2).astype(np.uint8)
        return img_arr
    

    def save(self, file):
        with open(file, "a") as f:
            f.write(str(self.points)+'\n'+
                    str(self.c)+'\n')
    

    def read(self, file, index):
        with open(file, "r") as f:
            d = f.readlines()
            d = d[index*2:(index+1)*2]
            for s in d[0][1:-2].split('), '):
                if s == '':
                    continue
                s = s.replace('(', '')
                s = s.replace(')', '')
                self.points.append(tuple([float(i) for i in s.split(',')]))
            self.c = tuple([int(i) for i in d[1].replace('(', '').replace(')', '').split(', ')])
