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
    [xs, ys] = np.random.rand(2, 2)  # 2 random support points "s"
    [a, b] = line_param(xs, ys)

    plt.figure(1)
    plt.scatter(xs, ys, c='r')  # support points in red
    plt.axis('equal')

    x = np.random.rand(n)  # n random points -> x coordinates
    y = a * x + b          # n random points -> y coordinates

    radius = 3
    colors = np.random.rand(n)
    area = np.pi * (radius * np.ones(n)) ** 2

    x_min = np.concatenate((np.zeros(1), xs, x), axis=0).min()
    x_max = np.concatenate((np.ones(1),  xs, x), axis=0).max()
    y_min = np.concatenate((np.zeros(1), ys, y), axis=0).min()
    y_max = np.concatenate((np.ones(1),  ys, y), axis=0).max()
    [xlim, ylim] = line_limits(a, b, x_min, x_max, y_min, y_max)

    plt.plot(xlim, ylim, 'g--', lw=2)  # limits dash line
    # plt.scatter(x, y, s=area, c=colors, alpha=0.5)

    # plt.plot(np.linspace(xlim[0], xlim[1], 20), ylim[0] * np.ones(20), 'r--', lw=2)
    # plt.plot(np.linspace(xlim[0], xlim[1], 20), ylim[1] * np.ones(20), 'r--', lw=2)
    # plt.plot(xlim[0] * np.ones(20), np.linspace(ylim[0], ylim[1], 20), 'r--', lw=2)
    # plt.plot(xlim[1] * np.ones(20), np.linspace(ylim[0], ylim[1], 20), 'r--', lw=2)

    h_max = math.sqrt((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2)  # euclidean distance of the 2 extreme points

    ratio = 0.20  # width / height ratio
    w_max = ratio * h_max

    w_mu = 0
    w_sigma = w_max / 3
    width_ = w_sigma * np.random.randn(n) + w_mu
    xdata = x + width_ * (-a) / math.sqrt(a ** 2 + 1)   # x data coordinates
    ydata = y + width_ * 1 / math.sqrt(a ** 2 + 1)      # y data coordinates
    plt.scatter(xdata, ydata, s=area, c=colors, alpha=0.5)

    # for i in range(n):
    #     point_projection(xd[i], yd[i], a, b)

    xlim2 = [min(xdata), max(xdata)]
    ylim2 = [min(ydata), max(ydata)]

    return a, b, xdata, ydata, xlim2, ylim2


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
    #
    # mu = 6
    # sigma = 1
    # generate_noise(n, mu, sigma)

    n = 20
    [a, b, xd, yd, xlim, ylim] = generate_noisy_line(n)

    # Least squares - dummy solution : A.x = c <=> x = (A^T*A)^-1.A^T.c
    A = np.concatenate((xd.reshape((n, 1)), np.ones((n, 1))), axis=1)
    c = yd.reshape((n, 1))
    [a_, b_] = np.linalg.inv(A.transpose()@A) @ (A.transpose() @ c)  # pseudo-inverse

    [x_, y_] = line_limits(a_[0], b_[0], xlim[0], xlim[1], ylim[0], ylim[1])
    plt.plot(x_, y_, 'b--', lw=2)
    plt.show()

    # remove the model: eps_i = y - a x_i - b
