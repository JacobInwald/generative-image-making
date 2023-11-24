from draw import *

class quad_manager:

    def __init__(self, n, tgt, brushsize, quad_type=quad_point, metric=METRIC.psnr):
        self.w = tgt.width
        self.h = tgt.height
        self.tgt = tgt
        self.quad_type = quad_type
        self.generation = {quad_type.gen_rand_quad(self.w,self.h,brushsize):0
                        for i in range(n)}
        self.n = n
        self.brushsize = brushsize
        self.metric = metric
        self.order = metric == METRIC.ncc or \
                     metric == METRIC.psnr

    def next_gen(self,randchild=False, randchildpercent=0.5, tolerance=0.01):
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
            self.generation[k] = self.tgt.pix_diff(k.draw(im), self.metric)
        self.generation = {k: v for k, v in sorted(self.generation.items(), key=lambda item: item[1], reverse=self.order)}


    def find_winner(self, n: int, cur: img):
        self.next_gen(True, randchildpercent=1)
        i = 0
        restartcount = 0
        loopcount = 0
        while i < n:
            loopcount +=1
            self.score_gen(cur)
            self.next_gen(randchild=True, randchildpercent=0.25)
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
                      quad_type=quad_point,
                      metric='psnr'):
    target = img(tgt_path, from_img=True)
    generation = quad_manager(gen_size, target, brushsize, quad_type, metric)
    c = canvas(can,target.width,target.height)
    cur = c.to_img()
    for i in range(num_quads):
        q,s_q = generation.find_winner(gen_iter, cur)
        if q is None:   break
        print(s_q)
        c.shapes.append(q)
        cur = c.to_img()
        print('Total score: ' + str(cur.pix_diff(target.pixel_map, metric)))
    c.save()

run_n_generations(100, 1000, 20, metric=METRIC.euc, tgt_path="george.png", can='canvas', quad_type=quad_point)