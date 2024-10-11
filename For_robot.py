import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si
import pandas as pd


def bezier_curve_rational(def_points, weight, speed=0.1):  # WARNING: May run into floating-point issues
    # def_points se zadávají jako list [[x1,y1],[x2,y2],[x3,y3]]
    n = len(def_points)
    points = []
    for t in [_ * speed for _ in range(int((1 + speed * 2) // speed))]:  # get values between 0 and 1
        # bezierova křivka
        points.append(
            [(sum(math.comb(n - 1, i) * ((1 - t) ** (n - 1 - i)) * (t ** i) * def_points[i][0] * weight[i] for i in
                  range(n))
              / sum(math.comb(n - 1, i) * ((1 - t) ** (n - 1 - i)) * (t ** i) * weight[i] for i in range(n))),
             (sum(math.comb(n - 1, i) * ((1 - t) ** (n - 1 - i)) * (t ** i) * def_points[i][1] * weight[i] for i in
                  range(n)) /
              sum(math.comb(n - 1, i) * ((1 - t) ** (n - 1 - i)) * (t ** i) * weight[i] for i in range(n)))])
    return points


def bezier_curve_explicit(def_points, speed=0.1):  # WARNING: May run into floating-point issues
    # def_points se zadávají jako list [[x1,y1],[x2,y2],[x3,y3]]
    n = len(def_points)
    points = []
    for t in [_ * speed for _ in range(int((1 + speed * 2) // speed))]:  # get values between 0 and 1
        # Bernsteinův polynom
        points.append(
            [sum(math.comb(n - 1, i) * ((1 - t) ** (n - 1 - i)) * (t ** i) * def_points[i][0] for i in range(n)),
             sum(math.comb(n - 1, i) * ((1 - t) ** (n - 1 - i)) * (t ** i) * def_points[i][1] for i in range(n))])
    return points


def bspline(cv, degree, n=100, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree, 1, degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree, 1, count - 1)

    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0 - degree, count + degree + degree - 1, dtype='int')
    else:
        kv = np.concatenate(([0] * degree, np.arange(count - degree + 1), [count - degree] * degree))

    # Calculate query range
    u = np.linspace(periodic, (count - degree), n)

    # Calculate result
    return np.array(si.splev(u, (kv, cv.T, degree))).T


x = [5655, 5668, 6021, 5998]
y = [865, 1018, 1018, 875]
vaha = [1, 0.3, 0.3, 1]
objekt = np.array([[5837+50, 865], [5837+50, 937+40], [5855+70, 937+20], [5855+70, 865]])
points = []
bez_len, bspl_len, base_len = 0, 0, 0

for i in range(len(x)):
    points.append([x[i], y[i]])

bezier = np.array(bezier_curve_rational(points, vaha))
B_spline = bspline(points, 2)

x_spline = B_spline[:, 0]
y_spline = B_spline[:, 1]

x_bezier = bezier[:, 0]
y_bezier = bezier[:, 1]

df = pd.DataFrame(bezier)
## save to xlsx file
filepath = 'my_bezier.xlsx'
df.to_excel(filepath, index=False)


for i in range(1, len(x_spline)):
    bspl_len += math.sqrt((x_spline[i] - x_spline[i - 1]) ** 2 + (y_spline[i] - y_spline[i - 1]) ** 2)

for i in range(1, len(x_bezier)):
    bez_len += math.sqrt((x_bezier[i] - x_bezier[i - 1]) ** 2 + (y_bezier[i] - y_bezier[i - 1]) ** 2)

for i in range(1, len(x)):
    base_len += math.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)

print(
    f"B-spline má délku {bspl_len:0.2f}, bézier má délku {bez_len:0.2f} a původní trajektorie mezi body je {base_len:0.2f}")


plt.title("Neoptimalizovaná trajektorie")
plt.xlabel("Osa X")
plt.ylabel("Osa Z")
plt.plot(x, y, 'ro', label="original")
plt.plot(x, y, 'b', label="linear interpolation")
plt.plot(x_spline, y_spline, '#2ED71C', label="B-spline(2)")
plt.plot(x_bezier, y_bezier, '#CF49FA', label="Bézier curve")
plt.plot(objekt[:, 0], objekt[:, 1], 'b')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid()
plt.savefig("Robot-aproximace")
plt.show()
