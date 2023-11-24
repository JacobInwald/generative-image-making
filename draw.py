from typing import Tuple
from PIL import Image
import numpy as np
import random as r
import cv2
import colorsys
from skimage.metrics import structural_similarity as ssim

def random_rgb():
    hue = r.uniform(0.0, 1.0)
    saturation = r.uniform(0.4, 1.0)
    value = r.uniform(0.4, 1.0)
    a = r.randint(1,255)

    rd, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    rd = max(int(rd * 255), 1)
    g = max(int(g * 255), 1)
    b = max(int(b * 255), 1)

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
    
    def pix_diff(self, pxlmap, subsample=None):
        pxlmap = pxlmap[:, :, :3]
        this_pxlmap = self.pixel_map[:, :, :3]
        if subsample is not None:
            # Subsample the pixel maps
            pxlmap = pxlmap[::subsample, ::subsample, :]
            this_pxlmap = this_pxlmap[::subsample, ::subsample, :]

        # Compute the mean squared error
        mse = np.mean((pxlmap - this_pxlmap) ** 2)
        
        # # Compute the peak signal-to-noise ratio
        # psnr = 10 * np.log10(255 ** 2 / mse)
        return mse
    

    def save(self):
         Image.fromarray(self.pixel_map).save(self.path)


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
    
    def combine(self, other, noise=0):
        coeff = r.uniform(0.25,0.75)
        tl      = tuple( int(coeff * self.tl[i] + (1 - coeff) * other.tl[i] + noise*r.gauss(-1, 1))
                            for i in range(len(self.tl)))
        alpha   = max(min(coeff * self.alpha + (1 - coeff) * other.alpha + noise*r.gauss(-1, 1), 45), -45)
        beta    = max(min(coeff * self.beta + (1 - coeff) * other.beta + noise*r.gauss(-1, 1), 45), -45)
        gamma   = max(min(coeff * self.gamma + (1 - coeff) * other.gamma + noise*r.gauss(-1, 1), 45), -45)
        hyp     = int(coeff * self.hyp + (1 - coeff) * other.hyp + noise*r.gauss(-1, 1))
        c       = tuple( int(coeff * self.c[i] + (1 - coeff) * other.c[i] + noise*r.gauss(-1, 1))
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

        tl = (int(r.uniform(0,w-hyp)), int(r.uniform(0,h-hyp)))
        alpha = np.deg2rad(r.uniform(-45,45))
        beta = np.deg2rad(r.uniform(-45,45))
        gamma = np.deg2rad(r.uniform(-45,45))

        new = quad(tl, alpha, beta, gamma, hyp, random_rgb())
        return new
    

    # def gen_rand_quad2(w, h, brush_size):
    #     while True:
    #         # generate four random points
    #         points = [(int(r.uniform(0, w-1)), int(r.uniform(0, h-1))) for _ in range(4)]
            
    #         # calculate the area of the shape defined by the points
    #         area = 0.5 * abs((points[0][0] * points[1][1] + points[1][0] * points[2][1] +
    #                         points[2][0] * points[3][1] + points[3][0] * points[0][1]) -
    #                         (points[1][0] * points[0][1] + points[2][0] * points[1][1] +
    #                         points[3][0] * points[2][1] + points[0][0] * points[3][1]))
            
    #         # calculate the aspect ratio of the bounding box
    #         x_coords = [point[0] for point in points]
    #         y_coords = [point[1] for point in points]
    #         bounding_box_aspect_ratio = 1000 if max(y_coords) == min(y_coords) else (max(x_coords) - min(x_coords)) / (max(y_coords) - min(y_coords))
            
    #         # check if the area and aspect ratio meet the requirements
    #         if area >= brush_size and \
    #             min(bounding_box_aspect_ratio,
    #                  1/bounding_box_aspect_ratio) >= 1/3:
    #             new = quad(points[0],
    #                         points[1],
    #                         points[2],
    #                         points[3],
    #                 random_rgb(),
    #                  r.randint(0,255))
    #             new.move_inbounds(w,h)
    #             return new