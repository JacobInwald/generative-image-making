from draw import *
from queue import PriorityQueue
import heapq

class quad_manager:

    def __init__(self, n, tgt, brushsize, quad_type=quad, diff_type=IMAGE_DIFF_METHOD.psnr):
        self.w = tgt.width
        self.h = tgt.height
        self.tgt = tgt
        self.quad_type = quad_type
        self.generation = {quad_type.gen_rand_quad(self.w,self.h,brushsize):0
                        for i in range(n)}
        self.gen = PriorityQueue()
        self.n = n
        self.brushsize = brushsize
        self.diff_type = diff_type
        self.order = diff_type == IMAGE_DIFF_METHOD.ncc or \
                     diff_type == IMAGE_DIFF_METHOD.psnr

    def next_gen(self,randchild=False, randchildpercent=0.5, tolerance=0.01, cur=None):
        new_gen = [list(self.generation.keys())[i] for i in range(0,int(np.sqrt(self.n*(1-randchild*randchildpercent))))
                   if self.generation[list(self.generation.keys())[i]] >= tolerance]
        best = new_gen
        size = len(new_gen)
        if size >0:
            for i in range(size):
                for j in range(size):
                    if i != j and len(new_gen) < self.n:
                        new_gen.append(best[i].combine(best[j], noise=1))
            for i in range(int((self.n-len(new_gen)) / size)):
                for j in range(size):
                    new_gen.append(best[j].combine(self.quad_type.gen_rand_quad(self.w,self.h,self.brushsize), noise=0))
        else:
            for i in range(self.n):
                new_gen.append(self.quad_type.gen_rand_quad(self.w,self.h,self.brushsize))

        self.generation = {k:0 for k in new_gen}


    def score_gen(self, cur: img):
        pxlmap = cur.pixel_map
        for k in self.generation.keys():
            im = img(from_map=True, pxlmap=pxlmap)
            # k.draw(im)
            self.generation[k] = self.tgt.pix_diff(k.draw(im), self.diff_type)
        self.generation = {k: v for k, v in sorted(self.generation.items(), key=lambda item: item[1], reverse=self.order)}


    def find_winner(self, n: int, cur: img):
        early_break = [0,0]
        self.next_gen(True, randchildpercent=1,cur=cur)
        t = 0 
        c = canvas('winner', cur.width, cur.height)
        i = 0
        restartcount = 0
        loopcount = 0
        while i < n:
            if restartcount == 10:   return None, None
            loopcount +=1
            self.score_gen(cur)
            # if self.generation[list(self.generation.keys())[0]] <= 0.01:
            #     self.generation = {self.quad_type.gen_rand_quad(self.w,self.h,self.brushsize):0
            #             for i in range(self.n)}
            #     i=0
            #     restartcount+=1
            
            # early_break[t] = list(self.generation.keys())[0:int(np.sqrt(len(self.generation)))]
            # t = (t+1) % len(early_break)
            # if all((early_break[i] == early_break[i+1] for i in range(len(early_break)-1))) \
            #     and i > n/2:
            #     break
            self.next_gen(cur=cur, randchild=True, randchildpercent=0.25)
            i+=1

        self.score_gen(cur)
        print('restarted', restartcount, 'times')
        print('looped', loopcount, 'times')
        print('average prob restart per loop', restartcount/loopcount)
        print('score: ', self.generation[list(self.generation.keys())[0]])
        winner = list(self.generation.keys())[0]
        return winner,self.generation[winner] 


def run_n_generations(num_quads, gen_size, gen_iter, 
                      brushsize=2, 
                      tgt_path="test.png",
                      can="canvas",
                      quad_type=quad,
                      diff_type='psnr'):
    target = img(tgt_path, from_img=True)
    generation = quad_manager(gen_size, target, brushsize, quad_type, diff_type)
    c = canvas(can,target.width,target.height)
    cur = c.to_img()
    for i in range(num_quads):
        q,s_q = generation.find_winner(gen_iter, cur)
        if q is None:   break
        print(s_q)
        c.shapes.append(q)
        cur = c.to_img()
        print('Total score: ' + str(cur.pix_diff(target.pixel_map, diff_type)))
    c.save()

# c = canvas('canvas',33,32)
# c.to_img()
# time.sleep(1)
# c.shapes[0].move_inbounds(33,32)
# c.to_img()

run_n_generations(100, 1000, 40, diff_type=IMAGE_DIFF_METHOD.euc, tgt_path="george.png", can='canvas', quad_type=quad_point)