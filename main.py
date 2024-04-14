from PIL import Image, ImageTk
import tkinter as tk
import numpy as np


class Field:
    def __init__(self, ro, u, type, inlet, outlet, eq):
        self.inlet = inlet
        self.outlet = outlet
        self.ro = ro
        self.u = u
        self.type = type
        self.eq = eq


class Domain:
    def __init__(self, rows, columns):

        self.fields_array = [[None for _ in range(columns)] for _ in range(rows)]
        for i in range(rows):
            for j in range(columns):
                self.fields_array[i][j] = Field(ro=0.5, u=[0, 0], type="empty space", inlet=[0]*9,
                                                outlet=[0]*9,
                                                eq=[0]*9)


class LBM_Diffusion:
    def __init__(self, rows, columns, tau, weights):
        self.domain = Domain(rows, columns)
        self.img_matrix = np.zeros((rows, columns, 3), dtype=np.uint8)
        self.tau = tau
        self.rows = rows
        self.cols = columns
        self.weights = weights
        self.c = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
        self.fill_matrix()

    def fill_matrix(self):
        for x in range(self.rows):
            for y in range(self.cols):
                if y == 1 or x == 1 or y == self.cols - 2 or x == self.rows - 2:
                    self.domain.fields_array[x][y] = Field(ro=0.5, u=[0, 0], type="wall", inlet=[0]*9,
                                                           outlet=[0]*9, eq=[0]*9)
                    self.img_matrix[x, y] = [255, 0, 0]
                if y == 60 and (0 < x < (((self.rows) / 2) - 25) or ((self.rows) / 2) + 25 < x <
                                self.rows):
                    self.domain.fields_array[x][y] = Field(ro=0.5, u=[0, 0], type="wall", inlet=[0]*9,
                                                           outlet=[0]*9, eq=[0]*9)
                    self.img_matrix[x, y] = [255, 0, 0]
                if 1 < y < 60 and 1 < x < self.rows - 2:
                    self.domain.fields_array[x][y] = Field(ro=1, u=[0, 0], type="empty space", inlet=[0]*9,
                                                           outlet=[0]*9, eq=[0]*9)
                    self.img_matrix[x, y] = [255, 255, 255]

    def collision(self):
        for x in range(self.rows):
            for y in range(self.cols):
                field_xy = self.domain.fields_array[x][y]
                for i in range(9):
                    field_xy.outlet[i] = (field_xy.inlet[i] + 1.0 / self.tau * (field_xy.eq[i] - field_xy.inlet[i]))
                self.domain.fields_array[x][y] = field_xy

    def streaming(self):
        for x in range(self.rows):
            for y in range(self.cols):
                field_xy = self.domain.fields_array[x][y]
                if field_xy.type != "wall":
                    field_xy.inlet[0] = field_xy.outlet[0]
                    if y - 1 > 0:
                        field_xy_left = self.domain.fields_array[x][y - 1]
                        field_xy.inlet[1] = field_xy_left.outlet[1]
                    else:
                        field_xy.inlet[1] = field_xy.outlet[2]
                    if y + 1 < self.cols:
                        field_xy_right = self.domain.fields_array[x][y + 1]
                        field_xy.inlet[2] = field_xy_right.outlet[2]
                    else:
                        field_xy.inlet[2] = field_xy.outlet[1]
                    if x - 1 > 0:
                        field_xy_down = self.domain.fields_array[x - 1][y]
                        field_xy.inlet[4] = field_xy_down.outlet[4]
                    else:
                        field_xy.inlet[4] = field_xy.outlet[3]
                    if x + 1 < self.rows:
                        field_xy_up = self.domain.fields_array[x + 1][y]
                        field_xy.inlet[3] = field_xy_up.outlet[3]
                    else:
                        field_xy.inlet[3] = field_xy.outlet[4]
                    if x - 1 > 0 and y - 1 > 0:
                        field_xy_up_left = self.domain.fields_array[x - 1][y - 1]
                        field_xy.inlet[8] = field_xy_up_left.outlet[8]

                    else:
                        field_xy.inlet[8] = field_xy.outlet[6]

                    if x - 1 > 0 and y + 1 < self.cols:
                        field_xy_up_right = self.domain.fields_array[x - 1][y + 1]
                        field_xy.inlet[7] = field_xy_up_right.outlet[7]
                    else:
                        field_xy.inlet[7] = field_xy.outlet[5]

                    if x + 1 < self.rows and y - 1 > 0:
                        field_xy_down_left = self.domain.fields_array[x + 1][y - 1]
                        field_xy.inlet[5] = field_xy_down_left.outlet[5]
                    else:
                        field_xy.inlet[5] = field_xy.outlet[7]

                    if x + 1 < self.rows and y + 1 < self.cols:
                        field_xy_down_right = self.domain.fields_array[x + 1][y + 1]
                        field_xy.inlet[6] = field_xy_down_right.outlet[6]
                    else:
                        field_xy.inlet[6] = field_xy.outlet[8]

                    field_xy.ro = np.sum(field_xy.inlet)

                    pux = 0.0
                    puy = 0.0
                    for i in range(9):
                        pux += field_xy.inlet[i] * self.c[i][0]
                        puy += field_xy.inlet[i] * self.c[i][1]

                    if field_xy.ro != 0:
                        field_xy.u[0] = pux / field_xy.ro
                        field_xy.u[1] = puy / field_xy.ro
                    else:
                        field_xy.u[0] = 0
                        field_xy.u[1] = 0
                    self.domain.fields_array[x][y] = field_xy

    def equilibrium(self, weights):
        for x in range(self.rows):
            for y in range(self.cols):
                field_xy = self.domain.fields_array[x][y]
                for i in range(9):
                    field_xy.eq[i] = weights[i] * field_xy.ro * (1.0 + 3.0 * self.c[i][0] * field_xy.u[0]
                                                                 + self.c[i][1] * field_xy.u[1] + 4.5 *
                                                                 (self.c[i][0] * field_xy.u[0] + self.c[i][1] * field_xy.u[1])
                                                                 * (self.c[i][0] * field_xy.u[0] + self.c[i][1] * field_xy.u[1])
                                                                 - 1.5 * (field_xy.u[0]*field_xy.u[0] + field_xy.u[1] * field_xy.u[1]))
                self.domain.fields_array[x][y] = field_xy

    def update_view(self):
        for x in range(self.rows):
            for y in range(self.cols):
                field_xy = self.domain.fields_array[x][y]
                if field_xy.type == "wall":
                    continue
                if field_xy.u[0] > 0:
                    self.img_matrix[x][y] = np.round([255 * field_xy.u[0], 0, 0])
                else:
                    self.img_matrix[x][y] = np.round([0, 0, -255 * field_xy.u[0]])


    def one_iteration(self):
        self.equilibrium(self.weights)
        self.collision()
        self.streaming()
        self.update_view()


class Window:
    def __init__(self, master, rows, cols):
        self.master = master
        self.frame = tk.Frame(self.master, background="black", width=cols, height=rows)
        self.LBM = LBM_Diffusion(rows=rows, columns=cols, tau=1.0, weights=[4.0/9.0, 1.0/9.0,
            1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])
        self.canvas = tk.Canvas(master, width=cols, height=rows)
        self.canvas.pack()
        self.counter = 0
        self.master.after(50, self.simulate)

    def simulate(self):
        self.LBM.one_iteration()
        img = ImageTk.PhotoImage(image=Image.fromarray(self.LBM.img_matrix, 'RGB'))
        self.canvas.image = img
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.master.after(50, self.simulate)


def main():
    root = tk.Tk()
    root.title("LBM Diffusion Simulation")

    rows = 300
    cols = 300

    window = Window(root, rows, cols)

    root.mainloop()


if __name__ == "__main__":
    main()
