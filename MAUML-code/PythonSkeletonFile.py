#import files for reading csv (pandas), plotting graphs and animations (matplotlib/pylab)

import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')


#Grabs file info and places the data into a list which can be used to create
#the animation of the arm in this instance. Files have moved since the skeleton
#file so may need tweaking. Was merely an experimentation base to try out things
#in python and so is a little naive.


def get_data():
    raw_data_set_full = pd.read_csv("practice set for python.csv",
                                    header=4,
                                    usecols=[2, 3, 4, 13, 14, 15, 24, 25, 26])

    raw_data_set_full.dropna(how='any', inplace=True)

    raw_data_set_full.to_csv("practice for python na's removed.csv")

    raw_data_set_r_wrist = pd.read_csv("practice for python na's removed.csv",
                                       usecols=[1, 2, 3]
                                       )

    raw_data_set_r_elbow = pd.read_csv("practice for python na's removed.csv",
                                       usecols=[4, 5, 6]
                                       )

    raw_data_set_r_shoulder = pd.read_csv("practice for python na's removed.csv",
                                          usecols=[7, 8, 9]
                                          )

    r_wrist = raw_data_set_r_wrist.values
    r_elbow = raw_data_set_r_elbow.values
    r_shoulder = raw_data_set_r_shoulder.values

    x1 = []
    x2 = []
    x3 = []
    y1 = []
    y2 = []
    y3 = []
    z1 = []
    z2 = []
    z3 = []

    frames = len(r_wrist)
    for i in range(frames):
        x1.append(r_wrist.item((i, 0)))
        y1.append(r_wrist.item((i, 1)))
        z1.append(r_wrist.item((i, 2)))
        x2.append(r_elbow.item((i, 0)))
        y2.append(r_elbow.item((i, 1)))
        z2.append(r_elbow.item((i, 2)))
        x3.append(r_shoulder.item((i, 0)))
        y3.append(r_shoulder.item((i, 1)))
        z3.append(r_shoulder.item((i, 2)))

    data = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
    return data

#update is a function which is required when using the FuncAnimation method to animate a
#graph. Basically it picks the next sequence of plot points (or the next frame of the animation)
#depending upon how you look at it.


def update(num):
    ax.clear()
    ax.set_xlim3d(-250, 2250)
    ax.set_ylim3d(-1000, 1500)
    ax.set_zlim3d(-500, 2000)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter(d[0][0][num], d[0][1][num], d[0][2][num], s=20, depthshade=True)
    ax.scatter(d[1][0][num], d[1][1][num], d[1][2][num], s=20, depthshade=True)
    ax.scatter(d[2][0][num], d[2][1][num], d[2][2][num], s=20, depthshade=True)
    ax.plot([d[0][0][num], d[1][0][num], d[2][0][num]], [d[0][1][num], d[1][1][num], d[2][1][num]], [d[0][2][num], d[1][2][num], d[2][2][num]], color='r')


#creating the necessary variables for plotting graphs, first is the graph stage as a whole
#second is the set of axes which will in turn hold the points I plot onto it.

fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

d = get_data()

ani = animation.FuncAnimation(fig, update, frames=len(d[0][0])-2, interval=1)

plt.show()
