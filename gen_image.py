from draw import *
import time

class quad_manager:

    def __init__(self, n, tgt, brushsize):
        self.w = tgt.width
        self.h = tgt.height
        self.tgt = tgt
        self.generation = {quad.gen_rand_quad3(self.w,self.h,brushsize):0
                        for i in range(n)}
        self.n = n
        self.brushsize = brushsize

    def next_gen(self):
        new_gen = [list(self.generation.keys())[i] for i in range(0,int(np.sqrt(self.n)))]
        
        size = len(new_gen)
        for i in range(size):
            for j in range(size):
                if i != j:
                    new_gen.append(new_gen[i].combine(new_gen[j]))
        
        rand_babies = int(0.9*self.n)
        new_gen = new_gen[:-rand_babies]
        for i in range(rand_babies):
            new_gen.append(quad.gen_rand_quad3(self.w,self.h,self.brushsize))
        
        self.generation = {k:0 for k in new_gen}


    def score_gen(self, cur: img):
        pxlmap = cur.pixel_map
        # st = time.time()
        for k in self.generation.keys():
            im = img(from_map=True, pxlmap=pxlmap)
            comp = k.draw(im)
            self.generation[k] = self.tgt.pix_diff(comp)
        # print(time.time()-st)
        self.generation = {k: v for k, v in sorted(self.generation.items(), key=lambda item: item[1])}


    def find_winner(self, n: int, cur: img):
        # c = canvas('winner', cur.width, cur.height)
        for i in range(n):
            self.score_gen(cur)
            # c.shapes = [list(self.generation.keys())[0]]
            # im = c.to_img()
            # time.sleep(0.01)
            self.next_gen()
        
        self.score_gen(cur)
        return list(self.generation.keys())[0]


def run_n_generations(num_quads, gen_size, gen_iter, brushsize=10, tgt_path="test.png", can="canvas"):
    target = img(tgt_path, from_img=True)
    c = canvas(can,target.width,target.height)
    cur = c.to_img()
    for i in range(num_quads):
        generation = quad_manager(gen_size, target, brushsize)
        q = generation.find_winner(gen_iter, cur)
        c.shapes.append(q)
        cur = c.to_img()
    c.save()

run_n_generations(200, 1000, 15, brushsize=250, tgt_path="george.png", can='canv2')