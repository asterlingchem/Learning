import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x = np.array([0.21121121, -0.4034034, 0.39139139, 0.78578579, -0.55155155,
                   0.56556557, 0.92392392, -0.6996997, 0.95795796, -0.91591592])

    y = np.array([0.23533653, -0.31035679, 0.45911638, 0.31506996, -0.29502177,
                   0.60401143, -0.50146046, -0.04222024, -0.82460688, 1.27242725])

    def line(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d


    popt, pcov = curve_fit(line, x, y)
    print(popt)

    smooth_xs = np.linspace(-1., 1.)
    plt.plot(x, y, '.')
    plt.plot(smooth_xs, line(smooth_xs, popt[0], popt[1], popt[2], popt[3]))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()
