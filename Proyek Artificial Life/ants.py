import os
import math
import time
import numpy
import pandas
import random
import matplotlib
import numpy.random as nrand
import matplotlib.pylab as plt
from sklearn.preprocessing import normalize


class AntColonyOptimization:
    def __init__(self):
        pass


class Grid:
    def __init__(self, height, width, path, rand_test=True):
        #untuk inisialisasi 2D grid

        self.path = path
        #buat nyimpan grid
        self.dim = numpy.array([height, width])
        #inisialisasi matrix
        self.grid = numpy.empty((height, width), dtype=Datum)
        if rand_test:
            #mengisi grid scr random
            self.rand_grid(0.25)
        #draw plot
        plt.ion()
        plt.figure(figsize=(10, 10))
        self.max_d = 0.001

    def rand_grid(self, sparse):
        #untuk ngisi grid scr random
        #sparse : presentase grid yg diisi

        for y in range(self.dim[0]):
            for x in range(self.dim[1]):
                if random.random() <= sparse:
                    r = random.randint(0, 1)
                    if r == 0:
                        self.grid[y][x] = Datum(nrand.normal(5, 0.25, 10))
                    elif r == 1:
                        self.grid[y][x] = Datum(nrand.normal(-5, 0.25, 10))

    def matrix_grid(self):
        #grid mjd matrix

        matrix = numpy.empty((self.dim[0], self.dim[1]))
        matrix.fill(0)
        for y in range(self.dim[0]):
            for x in range(self.dim[1]):
                if self.grid[y][x] is not None:
                    matrix[y][x] = self.get_grid()[y][x].condense()
        return matrix

    def plot_grid(self, name="", save_figure=True):
        #gambar 2D dari grid

        plt.matshow(self.matrix_grid(), cmap="RdBl", fignum=0)
        # buat save image
        if save_figure:
            plt.savefig(self.path + name + '.png')
        plt.draw()

    def get_grid(self):
        return self.grid

    def get_probability(self, d, y, x, n, c):
        #ini untuk mendapat kemungkinan drop/pickup masing" swarm
        #D untuk swarm
        #x untuk lokasi x swarm
        #y untuk lokasi y swarm
        #n untuk size

        # lokasi awal x dan y
        y_s = y - n
        x_s = x - n
        total = 0.0
        # untuk yg berdekatan
        for i in range((n*2)+1):
            xi = (x_s + i) % self.dim[0]
            for j in range((n*2)+1):
                # jika berdekatan
                if j != x and i != y:
                    yj = (y_s + j) % self.dim[1]
                    # pergi ke tempat yg deket, o
                    o = self.grid[xi][yj]
                    # mendapat kesamaan dari 0 ke x
                    if o is not None:
                        s = d.similarity(o)
                        total += s
        md = total / (math.pow((n*2)+1, 2) - 1)
        if md > self.max_d:
            self.max_d = md
        density = total / (self.max_d * (math.pow((n*2)+1, 2) - 1))
        density = max(min(density, 1), 0)
        t = math.exp(-c * density)
        probability = (1-t)/(1+t)
        return probability


class Ant:
    def __init__(self, y, x, grid):
        #inisialisasi semut

        self.loc = numpy.array([y, x])
        self.carrying = grid.get_grid()[y][x]
        self.grid = grid

    def move(self, n, c):
        step_size = random.randint(1, 25)
        # menambah vektor (-1,+1) * step_size ke lokasi semut
        self.loc += nrand.randint(-1 * step_size, 1 * step_size, 2)
        # mod lokasi baru, agar tidak terjadi overflow
        self.loc = numpy.mod(self.loc, self.grid.dim)
        # mendapat object pada lokasi
        o = self.grid.get_grid()[self.loc[0]][self.loc[1]]
        # jika sudah ada cell, pindah lagi
        if o is not None:
            # jika semut sudah membawah objek
            if self.carrying is None:
                # cek apakah semut mengambil objek
                if self.p_pick_up(n, c) >= random.random():
                    # mengambil objek
                    self.carrying = o
                    self.grid.get_grid()[self.loc[0]][self.loc[1]] = None
                # jika belum ngambil objek, move
                else:
                    self.move(n, c)
            # jika sudah membawa objek, move
            else:
                self.move(n, c)
        # apabila dalam sel kosong
        else:
            if self.carrying is not None:
                # cek apakah semut menjatuhkan objek
                if self.p_drop(n, c) >= random.random():
                    # menjatuhkan objek pada lokasi kosong
                    self.grid.get_grid()[self.loc[0]][self.loc[1]] = self.carrying
                    self.carrying = None

    def p_pick_up(self, n, c):
        #kemungkinan untuk mengambil objek
       
        ant = self.grid.get_grid()[self.loc[0]][self.loc[1]]
        return 1 - self.grid.get_probability(ant, self.loc[0], self.loc[1], n, c)

    def p_drop(self, n, c):
        #kemungkinan menjatuhkan objek
        ant = self.carrying
        return self.grid.get_probability(ant, self.loc[0], self.loc[1], n, c)


class Datum:
    def __init__(self, data):
        self.data = data

    def similarity(self, datum):
        #jarak antar swarm(datum)
        diff = numpy.abs(self.data - datum.data)
        return numpy.sum(diff**2)

    def condense(self):

        return numpy.mean(self.data)


def optimize(height, width, ants, sims, n, c, freq=500, path="image"):
    
    # inisialisasi grid
    grid = Grid(height, width, path)
    # meciptakan semut
    ant_agents = []
    for i in range(ants):
        ant = Ant(random.randint(0, height - 1), random.randint(0, width - 1), grid)
        ant_agents.append(ant)
    for i in range(sims):
        for ant in ant_agents:
            ant.move(n, c)
        if i % freq == 0:
            print(i)
            s = "img" + str(i).zfill(6)
            grid.plot_grid(s)


if __name__ == '__main__':
    optimize(50, 50, 150, 100000, 25, 10, freq=500, path="images/")
