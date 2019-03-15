import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')


def parse_data():
    data = []
    incorrect_file = True
    while incorrect_file:
        try:
            file_name = input("Please enter the file you would like to load: ")
            raw_data_set_full = pd.read_csv("csv_files/" + file_name,
                                            header=4,
                                            usecols=[2, 3, 4,
                                                     13, 14, 15,
                                                     24, 25, 26,
                                                     35, 36, 37,
                                                     46, 47, 48,
                                                     57, 58, 59,
                                                     68, 69, 70,
                                                     79, 80, 81,
                                                     90, 91, 92,
                                                     101, 102, 103,
                                                     112, 113, 114,
                                                     123, 124, 125,
                                                     134, 135, 136,
                                                     145, 146, 147,
                                                     156, 157, 158,
                                                     167, 168, 169])

            raw_data_set_full.dropna(how='all', inplace=True)

            df = raw_data_set_full.values

            xs = []
            ys = []
            zs = []

            for i in range(len(df)):
                for j in range(0, 48, 3):
                    xs.append(df.item((i, j)))

            for i in range(len(df)):
                for j in range(1, 48, 3):
                    ys.append(df.item((i, j)))

            for i in range(len(df)):
                for j in range(2, 48, 3):
                    zs.append(df.item((i, j)))

            data = [xs, ys, zs]

        except FileNotFoundError:
            incorrect_file = True
            print("Incorrect file name, please try again...")
        else:
            incorrect_file = False

    return data


def add_lines(axes, x, y, z):
    axes.plot([x[3], x[2], x[0], x[4], x[5], x[6]],
              [y[3], y[2], y[0], y[4], y[5], y[6]],
              [z[3], z[2], z[0], z[4], z[5], z[6]], color='r')
    axes.plot([x[0], x[7], x[4]],
              [y[0], y[7], y[4]],
              [z[0], z[7], z[4]], color='r')
    axes.plot([x[0], x[8], x[4]],
              [y[0], y[8], y[4]],
              [z[0], z[8], z[4]], color='r')
    axes.plot([x[9], x[7], x[13]],
              [y[9], y[7], y[13]],
              [z[9], z[7], z[13]], color='r')
    axes.plot([x[9], x[8], x[13]],
              [y[9], y[8], y[13]],
              [z[9], z[8], z[13]], color='r')
    axes.plot([x[12], x[11], x[9], x[13], x[14], x[15]],
              [y[12], y[11], y[9], y[13], y[14], y[15]],
              [z[12], z[11], z[9], z[13], z[14], z[15]], color='r')


def add_lims_and_labels(axes):
    axes.clear()
    axes.set_xlim3d(-1500, 2500)
    axes.set_ylim3d(-1000, 1500)
    axes.set_zlim3d(0, 2000)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')


def update(num):
    global ind
    end = ind + 16
    x = xs[ind: end]
    y = ys[ind: end]
    z = zs[ind: end]
    # print("the xs given are: ", xs[ind:ind + 16])
    # print("the ys given are: ", ys[ind:ind + 16])
    # print("the zs given are: ", zs[ind:ind + 16])
    # print("the new value of num is: ", ind)
    if len(x) >= 16 and len(y) >= 16 and len(z) >= 16:
        add_lims_and_labels(ax)
        ax.scatter(x, y, z, s=20)
        add_lines(ax, x, y, z)
        ind += 16


ind = 0
d = parse_data()
xs = d[0]
ys = d[1]
zs = d[2]

fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ani = animation.FuncAnimation(fig, update, frames=(len(xs) // 16), interval=1)

plt.show()
