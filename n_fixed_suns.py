from __future__ import annotations

import os
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def time_taken(start, end):
    t = end - start
    s = t % 60
    m = (t // 60) % 60
    h = t // 3600

    t_str = f"{s:.3f}s"
    if m > 0:
        t_str = f"{m}m " + t_str
    if h > 0:
        t_str = f"{h}h " + t_str
    
    return t_str


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

    def draw_2d(self, ffwd=50_000, step=1, savefile='ani.mp4'):
        """Plot the trajectory of the current object

        Args:
            ffwd (int, optional): _description_. Defaults to 50_000.
            step (int, optional): _description_. Defaults to 1.
            savefile (str, optional): _description_. Defaults to 'ani.mp4'.
        """
        fig, ax = plt.subplots()
        start, end = 1, 1

        # Draw the suns
        for sun in self.suns:
            sun_x, sun_y = sun
            plt.scatter([sun_x], [sun_y], [50], c='red')  # third unnamed arg is size

        # Draw the planet's trajectory
        step = 1
        traj = np.array(self.history)
        traj_X, traj_Y = traj.T
        def planetary_motion(i):
            if start == 1:
                start *= time.time()
            s = (i - 1) * step * ffwd
            if s >= len(traj_X): return

            e = i * step * ffwd
            plt.plot(traj_X[s:e], traj_Y[s:e], 'k')

            # Draw the last planet location
            if e >= len(traj_X):
                if end == 1:
                    end *= time.time()
                my_x, my_y = self.body
                plt.scatter([my_x], [my_y], [10], c='blue')

        animator = ani.FuncAnimation(fig, planetary_motion, interval=step, save_count=(len(traj_X) // ffwd))
        animator.save(savefile, ani.FFMpegWriter(fps=60))
        plt.show()
        print(f"Animated {self.print()} in {time_taken(start, end)}")


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


def parse_cli_args():
    parser = argparse.ArgumentParser()

    # data gen args
    parser.add_argument('--mode', required=True, choices=('sim', 'spar', 'draw'), type=str)
    parser.add_argument('--xoffset', default=0, type=int)
    parser.add_argument('--yoffset', default=0, type=int)
    parser.add_argument('--clicks', default=130_000_000, type=int)
    parser.add_argument('--tag', default='', type=str)
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--root', default='.', type=str)
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--draw', action='store_true')

    # data sparser args
    parser.add_argument('--tags', type=str, nargs='+')
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--sparsity', default=50_000, type=int)
    parser.add_argument('--savefile', type=str)

    args = parser.parse_args()
    return args


@timeit
def data_gen(args):
    fname = f"dump{args.tag}.bin"
    fpath = os.path.join(args.root, fname)

    if args.load:
        start = time.time()
        with open(fpath, 'rb') as fh:
            sys = pickle.load(fh)
        end = time.time()
        print(f"Loaded {fname} in {time_taken(start, end)}")

    if not args.load:
        start = time.time()
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
        end = time.time()
        print(f"Generating starting point in {time_taken(start, end)}")

    if not args.load or args.cont:
        start = time.time()
        sys.ticktock(args.clicks)
        end = time.time()
        print(f"Simulated {args.clicks} steps in {time_taken(start, end)}")

    sys.print()

    if args.save:
        start = time.time()
        with open(fpath, 'wb') as fh:
            p = pickle.Pickler(fh)
            p.fast = True
            p.dump(sys)
        end = time.time()
        print(f"Saved simulation results in {time_taken(start, end)}")
    
    if args.draw:
        start = time.time()
        sys.draw_2d()
        end = time.time()


@timeit
def data_sparser(args):     # TODO: try keeping delta_r of <= 1e-2
    fnames = {f'dump{x}.bin': f'dump{x}_sp{args.sparsity}.bin' for x in args.tags}
    fpaths = { os.path.join(args.root, k): os.path.join(args.root, v) for k, v in fnames.items() }
    for in_fpath, out_fpath in fpaths.items():
        start = time.time()
        with open(in_fpath, 'rb') as fh:
            sys = pickle.load(fh)
        end = time.time()
        print(f"Loaded {in_fpath} in {time_taken(start, end)}")
        
        start = time.time()
        sys.history = sys.history[::args.sparsity]
        end = time.time()
        print(f"Sparsified data in {time_taken(start, end)}")

        start = time.time()
        with open(out_fpath, 'wb') as fh:
            p = pickle.Pickler(fh)
            p.fast = True
            p.dump(sys)
        end = time.time()
        print(f"Saved {out_fpath} in {time_taken(start, end)}")


def orbit_artist(args):
    fig, ax = plt.subplots()
    are_suns_drawn = False
    fnames = [f'dump{x}_sp{args.sparsity}.bin' for x in args.tags]
    fpaths = [os.path.join(args.root, x) for x in fnames]

    trajs = []
    start = time.time()
    for fpath in fpaths:
        with open(fpath, 'rb') as fh:
            sys = pickle.load(fh)

            if not are_suns_drawn:
                # Draw the suns
                for sun in sys.suns:
                    sun_x, sun_y = sun
                    plt.scatter([sun_x], [sun_y], [50], c='black')  # third unnamed arg is size
                are_suns_drawn = True

            sys.history = np.array(sys.history).T
            trajs.append(sys.history)
    end = time.time()
    print(f"Loaded input files in {time_taken(start, end)}")
    colors = ['lime', 'cadetblue', 'sienna']
    inv_colors = ['orange', 'red', 'lightsalmon']

    if args.animate:
        longest_path = max(len(x) for x, _ in trajs)
        fps = 60
        print(f"Preparing to animate {longest_path} frames at {fps}")

        colors = ['silver', 'cyan', 'orange']
        inv_colors = ['gold', 'red', 'blue']

        def planetary_motions(i):
            if i > longest_path: return

            for idx, traj in enumerate(trajs):
                # Draw the last planet location
                if i * 2 >= len(traj[0]):
                    plt.scatter([traj[0][-1]], [traj[1][-1]], [10], c=colors[idx % len(inv_colors)])
                    continue

                plt.plot(traj[0][i : i+2], traj[1][i : i+2], 'k') #colors[idx % len(colors)])

        start = time.time()
        animator = ani.FuncAnimation(fig, planetary_motions, interval=1, save_count=(longest_path // 2 + 1))
        if args.savefile: animator.save(args.savefile, ani.FFMpegWriter(fps=fps))
        plt.show()
        end = time.time()
        print(f"Animated orbitals in {time_taken(start, end)}")

    else:
        for idx, traj in enumerate(trajs):
            plt.plot(traj[0], traj[1], colors[idx % len(colors)], linewidth=1)
            plt.scatter([traj[0][-1]], [traj[1][-1]], [10], c=inv_colors[idx % len(inv_colors)])

        plt.show()


if __name__ == '__main__':
    args = parse_cli_args()
    if args.mode == 'spar':
        data_sparser(args)
    elif args.mode == 'draw':
        orbit_artist(args)
    elif args.mode == 'sim':
        data_gen(args)
