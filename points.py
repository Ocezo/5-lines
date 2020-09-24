import sys
import math
import numpy as np
import matplotlib.pyplot as plt


def line_param(x2, y2):
    if len(x2) != 2 or len(y2) != 2:
        print('Dimension error!', file=sys.stderr)
    a_ = (y2[1] - y2[0]) / (x2[1] - x2[0])
    b_ = y2[0] - a_ * x2[0]
    return a_, b_


def line_limits(a, b, x_min, x_max, y_min, y_max):
    xlim = []
    ylim = []

    if (y_min < a * x_min + b) & (a * x_min + b < y_max):  # y=ax+b & x = x_min
        xlim.append(x_min)
        ylim.append(a * x_min + b)

    if (y_min < a * x_max + b) & (a * x_max + b < y_max):  # y=ax+b & x = x_max
        xlim.append(x_max)
        ylim.append(a * x_max + b)

    if (x_min < (y_min - b) / a) & ((y_min - b) / a < x_max):  # x = (y-b)/a & y = y_min
        xlim.append((y_min - b) / a)
        ylim.append(y_min)

    if (x_min < (y_max - b) / a) & ((y_max - b) / a < x_max):  # x = (y-b)/a & y = y_max
        xlim.append((y_max - b) / a)
        ylim.append(y_max)

    return xlim, ylim


def generate_points(n):
    # generate n random points
    radius = 3
    colors = np.random.rand(n)
    [x_, y_] = np.random.rand(2, n)
    area = np.pi * (radius * np.ones(n)) ** 2

    plt.figure(1)
    plt.scatter(x_, y_, s=area, c=colors, alpha=0.5)  # random set
    plt.axis('equal')
    plt.show()

    return x_, y_


def generate_line():
    # generate 2 random points ~> line
    [xl, yl] = np.random.rand(2, 2)  # random line
    [a_, b_] = line_param(xl, yl)

    [x_min, x_max, y_min, y_max] = [0, 1, 0, 1]
    [xlim, ylim] = line_limits(a_, b_, x_min, x_max, y_min, y_max)

    plt.scatter(xl, yl)                # random extremities
    plt.plot(xlim, ylim, 'b--', lw=2)  # limits dash line

    return a_, b_


def generate_noisy_line(n):
    # generate 2 random points ~> line
    [xl, yl] = np.random.rand(2, 2)  # 2 random support points
    [a, b] = line_param(xl, yl)

    plt.figure(1)
    plt.scatter(xl, yl, c='r')  # support points in red
    plt.axis('equal')

    x_ = np.random.rand(n) # n random points -> x coordinates
    y_ = a * x_ + b        # n random points -> y coordinates

    radius = 3
    colors = np.random.rand(n)
    area = np.pi * (radius * np.ones(n)) ** 2

    x_min = np.concatenate((np.zeros(1), xl, x_), axis=0).min()
    x_max = np.concatenate((np.ones(1),  xl, x_), axis=0).max()
    y_min = np.concatenate((np.zeros(1), yl, y_), axis=0).min()
    y_max = np.concatenate((np.ones(1),  yl, y_), axis=0).max()
    [xlim, ylim] = line_limits(a, b, x_min, x_max, y_min, y_max)

    plt.plot(xlim, ylim, 'b--', lw=2)  # limits dash line
    # plt.scatter(x_, y_, s=area, c=colors, alpha=0.5)

    x_min = np.concatenate((xl, x_), axis=0).min()
    x_max = np.concatenate((xl, x_), axis=0).max()
    y_min = np.concatenate((yl, y_), axis=0).min()
    y_max = np.concatenate((yl, y_), axis=0).max()

    plt.scatter(x_min, y_min, c='orange')
    plt.scatter(x_max, y_max, c='orange')
    h_max = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)  # euclidean distance between the 2 extreme points

    ratio = 0.10  # width / height ratio
    w_max = ratio * h_max

    w_mu = 0
    w_sigma = w_max / 3
    width_ = w_sigma * np.random.randn(n) + w_mu
    xh_ = x_ + width_ * (-a) / math.sqrt(a ** 2 + 1)
    yh_ = y_ + width_ * 1 / math.sqrt(a ** 2 + 1)
    plt.scatter(xh_, yh_, s=area, c=colors, alpha=0.5)
    plt.show()

    # for i in range(n):
    #    point_projection(xh_[i], yh_[i], a, b)

    return a, b


def point_projection(x_m, y_m, a_, b_):
    # draw the M input point in magenta
    plt.scatter(x_m, y_m, c='m', alpha=0.75)

    # get the H projection point along y = a.x + b
    x_h = (x_m + a_ * (y_m - b_)) / (1 + a_ ** 2)
    y_h = b_ + a_ * (x_m + a_ * (y_m - b_)) / (1 + a_ ** 2)

    # draw the H projection point in orange
    plt.scatter(x_h, y_h, c='orange', alpha=0.75)

    # draw the orthogonal line between M and H in red
    plt.plot([x_m, x_h], [y_m, y_h], 'r--', lw=1.5)

    return x_h, y_h


def cloud_projection(x, y, a, b):
    xh = (x + a * (y - b)) / (1 + a ** 2)
    yh = b + a * (x + a * (y - b)) / (1 + a ** 2)
    plt.scatter(xh, yh, c='orange', alpha=0.5)
    plt.show()


def generate_noise(n, mu, sigma):
    x = sigma * np.random.randn(n) + mu

    x_pos = np.arange(n)
    plt.figure(2)
    plt.bar(x_pos, x, color='b')
    # plt.show()

    plt.figure(3)
    plt.hist(x, 50, color='c')
    # plt.show()


if __name__ == '__main__':
    # n = 100
    # [x, y] = generate_points(n)
    # [a, b] = generate_line()
    # cloud_projection(x, y, a, b)
    #
    # for i in range(n):
    #     point_projection(x[i], y[i], a, b)

    generate_noisy_line(20)

    # mu = 6
    # sigma = 1
    # generate_noise(n, mu, sigma)
