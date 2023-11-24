from typing import Tuple
from PIL import Image
from abc import ABC, abstractmethod
import numpy as np
import random as r
import cv2
from enum import Enum

# PIX DIFF TYPES
class METRIC(Enum):
    psnr    = 1
    mse     = 2
    ncc     = 3
    euc     = 4
    man     = 5


def random_rgb() -> Tuple[int,int,int,int]:
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
    
    
    def pix_diff(self, pxlmap:np.ndarray, diff_type:METRIC=METRIC.psnr, subsample:int=None)->float:
        ''' Takes in a pixel map and returns the difference between this image and the pixel using the 
            specified metric. subsample is an optional parameter that can be used to subsample the pixel maps'''
        pxlmap = pxlmap[:, :, :3]
        this_pxlmap = self.pixel_map[:, :, :3]
        if subsample is not None:
            # Subsample the pixel maps
            pxlmap = pxlmap[::subsample, ::subsample, :]
            this_pxlmap = this_pxlmap[::subsample, ::subsample, :]
        i1 = pxlmap 
        i2 = this_pxlmap

        # Compute the mean squared error
        if diff_type == METRIC.mse:
            return np.mean((i1 - i2) ** 2)
        # Compute the peak signal-to-noise ratio 
        elif diff_type == METRIC.psnr:
            return 10 * np.log10(255 ** 2 / np.mean((i1 - i2) ** 2))
        # Compute the normalized cross-correlation
        elif diff_type == METRIC.ncc:
            return np.sum( (i1 - np.mean(i1)) * (i2 - np.mean(i2)) ) / ((i1.size - 1) * np.std(i1) * np.std(i2) )
        # Compute the euclidean distance
        elif diff_type == METRIC.euc:
            return np.sqrt(np.sum((i1 - i2)^2)) / i1.size
        # Compute the manhattan distance
        elif diff_type == METRIC.man:
            return np.sum(abs(i1 - i2)) / i1.size

        return np.sum(abs(i1 - i2)) / i1.size

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
    

class quad(ABC):
    ''' Abstract class for a quad. A quad is short for quadrilateral here'''

    def draw(self, img: img):
            ''' Draws the quad onto the image and returns the image'''
            img_arr = img.pixel_map
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
            poly_pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(mask, [poly_pts], color=self.c)
            overlay = img_arr[mask > 0]
            img_arr[mask > 0] = ((overlay + self.c)/2).astype(np.uint8)
            return img_arr


    @abstractmethod
    def gen_rand_quad(w:int, h:int, brush_size:int):
        ''' Generates a random quad'''
        pass

    @abstractmethod
    def combine(self, other, noise:int=10):
        ''' Combines this quad with another quad and returns the result'''
        pass

    @abstractmethod
    def save(self, file:str):
        ''' Saves the quad to a file'''
        pass

    @abstractmethod
    def read(self, file:str, index:int):
        ''' Reads the quad from a file'''
        pass


class quad_vector(quad):

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
        self.points = [self.tl, self.tr, self.br, self.bl]

    
    def combine(self, other, noise:int=10):
        # Combines all the fields of the two quads with a random weight on one of the quads
        # and a random noise on top
        coeff   = r.gauss(0.5,0.5)
        tl      = tuple( max(int(coeff * self.tl[i] + (1 - coeff) * other.tl[i] + noise*r.gauss(0, 1)), 0)
                            for i in range(len(self.tl)))
        alpha   = max(min(coeff * self.alpha + (1 - coeff) * other.alpha + noise*r.gauss(-1, 1), 0.5), -0.5)
        beta    = max(min(coeff * self.beta + (1 - coeff) * other.beta + noise*r.gauss(-1, 1), 0.5), -0.5)
        gamma   = max(min(coeff * self.gamma + (1 - coeff) * other.gamma + noise*r.gauss(-1, 1), 0.5), -0.5)
        hyp     = max(int(coeff * self.hyp + (1 - coeff) * other.hyp + noise*r.gauss(0, 1)), 5)
        c       = tuple(min(255, max(int(coeff * self.c[i] + (1 - coeff) * other.c[i] + noise*r.gauss(0, 1)), 1))
                    for i in range(len(self.c)))

        return quad_vector(tl,alpha,beta,gamma,hyp,c)


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
            new = quad_vector(self.tl, self.alpha,self.beta,self.gamma,self.hyp,self.c)
            self.tr = new.tr
            self.bl = new.bl
            self.br = new.br


    def gen_rand_quad(w, h, brush_size):

        hyp = np.sqrt(brush_size)
        hyp += r.random()*(min(w,h)-hyp)
        hyp = int(hyp)
        hyp = min(hyp, min(w,h))

        tl = (int(r.uniform(0,w-hyp)), int(r.uniform(0,h-hyp)))
        alpha = r.uniform(-0.5,0.5)
        beta = r.uniform(-0.5,0.5)
        gamma = r.uniform(-0.5,0.5)

        new = quad_vector(tl, alpha, beta, gamma, hyp, random_rgb())
        return new


class quad_point(quad):
    ''' quad class but defined with points instead of angles and lengths. 
        The points are defined in clockwise order starting from the top left point'''
    
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
