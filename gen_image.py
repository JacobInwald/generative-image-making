from draw import *
import time

class quad_manager:

    def __init__(self, n, tgt, brushsize):
        self.w = tgt.width
        self.h = tgt.height
        self.tgt = tgt
        self.generation = {quad.gen_rand_quad(self.w,self.h,brushsize):0
                        for i in range(n)}
        self.n = n
        self.brushsize = brushsize

    def next_gen(self):
        new_gen = [list(self.generation.keys())[i] for i in range(0,int(np.sqrt(self.n)))]
        rand_babies = int(0.2*self.n)
        size = len(new_gen)
        num_new = size
        for i in range(size):
            if num_new+rand_babies >= self.n:
                    break
            for j in range(size):
                if num_new+rand_babies >= self.n:
                    break
                if i != j:
                    num_new += 1
                    new_gen.append(new_gen[i].combine(new_gen[j], noise=5))

        for i in range(rand_babies):
            new_gen.append(quad.gen_rand_quad(self.w,self.h,self.brushsize))
        
        
        self.generation = {k:0 for k in new_gen}


    def score_gen(self, cur: img, maximise_metric=True):
        pxlmap = cur.pixel_map
        for k in self.generation.keys():
            im = img(from_map=True, pxlmap=pxlmap)
            comp = k.draw(im)
            self.generation[k] = self.tgt.pix_diff(comp)
        self.generation = {k: v for k, v in sorted(self.generation.items(), key=lambda item: item[1], reverse=maximise_metric)}


    def find_winner(self, n: int, cur: img):
        early_break = [0,0,0,0,0,0]
        t = 0 
        # ? c = canvas('winner', cur.width, cur.height)
        for i in range(n):
            self.score_gen(cur, False)
            # ! early_break[t] = list(self.generation.keys())[0:6]
            # ! t = (t+1) % len(early_break)
            # ! if all((early_break[i] == early_break[i+1] for i in range(len(early_break)-1))) \
            # !     and i > n/2:
            # !     break
            # ? c.shapes = list(self.generation.keys())[0:5]
            # ? im = c.to_img()
            # ? time.sleep(1)
            self.next_gen()

        self.score_gen(cur, False)
        winner = list(self.generation.keys())[0]
        return winner,self.generation[winner] 


def run_n_generations(num_quads, gen_size, gen_iter, brushsize=10, tgt_path="test.png", can="canvas"):
    target = img(tgt_path, from_img=True)
    c = canvas(can,target.width,target.height)
    cur = c.to_img()
    for i in range(num_quads):
        generation = quad_manager(gen_size, target, brushsize)
        q,s_q = generation.find_winner(gen_iter, cur)
        print(s_q)
        c.shapes.append(q)
        cur = c.to_img()
    c.save()

# c = canvas('canvas',40,40)
# c.shapes.append(quad.gen_rand_quad(40,40,2))
# c.to_img()
# time.sleep(1)
# c.shapes.append(quad.gen_rand_quad(40,40,2))
# c.to_img()
# time.sleep(1)
# c.shapes.append(c.shapes[0].combine(c.shapes[1]))
# c.shapes = [c.shapes[-1]]
# c.to_img()

run_n_generations(100, 1000, 20, brushsize=250, tgt_path="george.png", can='canvas')