from __future__ import annotations

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani


class NSolarSystem:
    g = np.float64(9.8)

    def __init__(self, s, v, arr_of_nd_pts) -> None:
        if arr_of_nd_pts.shape[1] != s.shape[0] or s.shape[0] != v.shape[0]:
            raise ValueError("Dimension mismatch")

        self.suns = arr_of_nd_pts
        self.body = s
        self.vel = v

        self.history = []

    @staticmethod
    def rand_init(s=np.array((0,0), dtype=np.float64), v=np.array((0,0), dtype=np.float64), n=2, d=2) -> NSolarSystem:
        return NSolarSystem(s, v, np.random.random(n * d).reshape((n, d)) * 20 - 10)

    def print(self):
        print(self.body, self.vel)

    def tick(self, del_t=np.float64(1e-6)):
        # Add last position to history
        self.history.append(self.body.copy())

        # Find distance vectors from all suns
        D = self.suns - self.body

        # Square the distance vectors
        mag_D = np.linalg.norm(D, axis=1)
        mag_D = mag_D.reshape((len(mag_D), 1))
        unit_D = D / mag_D

        # Add up inverse vectors and scale by g
        inv_sq = unit_D / mag_D / mag_D
        del_v = (inv_sq).sum(axis=0)
        del_v *= NSolarSystem.g * del_t

        # Update the position with current v
        self.body += self.vel

        # Update the velocity with acceleration vector
        self.vel += del_v * del_t

    def ticktock(self, n, **kwargs):
        bound = 20
        for _ in range(n):
            if self.body.max() > bound or self.body.min() < -bound:
                break
            self.tick(**kwargs)

    def draw_2d(self, savefile='ani.mp4'):
        fig, ax = plt.subplots()

        # Draw the suns
        for sun in self.suns:
            sun_x, sun_y = sun
            plt.scatter([sun_x], [sun_y], [50], c='red')  # third unnamed arg is size

        # Draw the planet's trajectory
        step = 1
        ffwd = 50_000
        traj = np.array(self.history)
        traj_X, traj_Y = traj.T
        def planetary_motion(i):
            s = (i - 1) * step * ffwd
            e = i * step * ffwd
            plt.plot(traj_X[s:e], traj_Y[s:e], 'k')
            
            # Draw the last planet location
            if e >= len(traj_X):
                my_x, my_y = self.body
                plt.scatter([my_x], [my_y], [10], c='blue')

        animator = ani.FuncAnimation(fig, planetary_motion, interval=step, save_count=(len(traj_X) // ffwd))
        animator.save(savefile, ani.FFMpegWriter(fps=60))
        plt.show()


def timeit(f):
    import time

    def m(*args, **kwargs):
        start = time.perf_counter()
        res = f(*args, **kwargs)
        end = time.perf_counter()
        
        t = end - start
        print(f"Elapsed time: {(t // 60)}m {(t % 60):.3f}s")
        return res

    return m

@timeit
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xoffset', default=0, type=int)
    parser.add_argument('--yoffset', default=0, type=int)
    parser.add_argument('--clicks', default=130_000_000, type=int)
    parser.add_argument('--tag', default='', type=str)
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--multidraw', type=int, nargs='+')
    args = parser.parse_args()

    fname = f"dump{args.tag}.bin"

    if args.load:
        with open(fname, 'rb') as fh:
            sys = pickle.load(fh)

    if not args.load:
        if args.seed:
            with open("seed.bin", 'rb') as fh:
                wiggle = np.float64(1e-6)
                sys = pickle.load(fh)
                s_x, s_y = sys.body
                s_x += wiggle * args.xoffset
                s_y += wiggle * args.yoffset
                sys.body = np.array([s_x, s_y], dtype=np.float64)
        else:
            sys = NSolarSystem.rand_init(v=np.array([1e-6, 1e-6]), n=3)

    if not args.load or args.cont:
        sys.ticktock(args.clicks)

    sys.print()

    if args.save:
        with open(fname, 'wb') as fh:
            p = pickle.Pickler(fh)
            p.fast = True
            p.dump(sys)
    
    if args.draw:
        sys.draw_2d()


def scratch_main():
    with open("seed.bin", 'rb') as fh:
        sys = pickle.load(fh)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
    # scratch_main()
