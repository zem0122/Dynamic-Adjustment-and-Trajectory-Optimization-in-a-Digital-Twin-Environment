import math
import matplotlib as plt
import pandas as pd
import numpy as np
import scipy.interpolate as si


def fc_delka(body):
    e = 0
    for a in range(1, len(body)):
        e += math.sqrt((body[a, 0] - body[a - 1, 0]) ** 2 + (body[a, 1] - body[a - 1, 1]) ** 2)
    return e


def bezier_curve_rational(def_points, weight, speed=0.01):  # WARNING: May run into floating-point issues
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


def bezier_curve_explicit(def_points, speed=0.01):  # WARNING: May run into floating-point issues
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


def plotting(data, colour, label, plot):
    x, y = data[:, 0], data[:, 1]
    plot.plot(x, y, colour)
    plot.set_title(label)
    return None


def control(pod_jedna, objekt):
    for j in pod_jedna:
        if objekt[0, 0] < j[0] < objekt[3, 0] and (j[1] < objekt[1, 1] or j[1] < objekt[2, 1]):
            return False
    return True


def self_control(base_points, objekt, value):
    """ Tato funkce způsobuje samostatné vyhledávání nejlepší trajektorie s ohledem na zadanou překážku v prostoru a
    samovolně poté upravuje jednotlivé váhy bodů pokud zjistí že se část vygenerované trajektorie nachází v překážce"""
    control = []
    point = np.array(bezier_curve_rational(base_points, value))  # Generování bodů racionální bézierovy křivky
    for j in range(len(point)):
        if objekt[0, 0] <= point[j, 0] <= objekt[3, 0]:
            control.append(point[j, 1])  # Přidávání bodů do listu které jsou v rozmezí X osy objektu
    while min(control) < objekt[1, 1]:
        offset1, offset2 = 0, 0
        length = [[], []]
        control = []  # Nové vygenerování bodů v každé smyčce při hledání trajektorie bez srážky
        point = np.array(bezier_curve_rational(base_points, value))
        for j in range(len(point)):
            # Přidávání bodů do listu které jsou v rozmezí X osy objektu
            if objekt[0, 0] <= point[j, 0] <= objekt[2, 0]:
                control.append(point[j, 1])
            # Vstupní dotek z leva když je níž než objekt (pouze jeden bod)
            if point[j - 1, 0] < objekt[0, 0] < point[j, 0] and point[j, 1] < objekt[1, 1]:
                for k in base_points[1:len(base_points) - 1]:
                    if k[0] < point[j, 0]:
                        offset1 = base_points.index(k)
                        length[0].append(math.sqrt((k[0] - point[j, 0]) ** 2 + (k[1] - point[j, 1]) ** 2))
            # Výstupní dotek z prava když je níž než objekt (pouze jeden bod)
            if point[j - 1, 0] <= objekt[2, 0] <= point[j, 0] and point[j, 1] < objekt[2, 1]:
                for k in base_points[1:len(base_points) - 1]:
                    if k[0] > point[j, 0]:
                        length[1].append(math.sqrt((k[0] - point[j, 0]) ** 2 + (k[1] - point[j, 1]) ** 2))
                    else:
                        offset2 = base_points.index(k)  # Offset pro indexování výstupního bodu
        if len(length[0]) == 0 and len(length[1]) == 0:
            print(f"Délka dráhy = {fc_delka(point):.2f} a váha bodů = {value}")
            return point
        if len(length[0]) != 0 and len(length[1]) != 0:
            value[offset1] += 0.1
            value[offset2 + 1] += 0.1
        elif len(length[0]) != 0 and len(length[1]) == 0:
            value[offset1] += 0.1
        elif len(length[1]) != 0 and len(length[0]) == 0:
            value[offset2 + 1] += 0.1



# x,y z boxu nad pravý = [5655, 5668, 6021, 5998][865, 1018, 1018, 875]
# x,y z boxu na levý = [5655, 5668, 5311, 5303][865, 1018, 1018, 875]
x = [5655, 5668, 6021, 5998]
y = [865, 1018, 1018, 875]
vaha1 = [1, 1, 1, 1]
vaha2 = [1, 1, 1, 1]
figure, axis = plt.subplots(2, 2, figsize=(12, 8))
ax = [axis[0, 0], axis[0, 1], axis[1, 0], axis[1, 1]]
points = []
objekt = np.array([[5837+50, 865], [5837+50, 937+40], [5855+70, 937+20], [5855+70, 865]])
# Objekt vpravo - [[5837, 865], [5837, 937], [5855, 937], [5855, 865]]
# Objekt vlevo - [[5837-382, 865], [5837-382, 937], [5855-382, 937], [5855-382, 865]]
# Objekt vpravo jiný tvar - [[5837+50, 865], [5837+50, 937+40], [5855+70, 937+20], [5855+70, 865]]
vysledne = np.array([[0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0]])
vracene = [0, 0]
delka, nejkratsi1 = 0, [0, 0]
nejkratsi2 = [0, 0]
nejkratsi3 = [0, 0]

rozhodnuti = False

for i in range(len(x)):
    points.append([x[i], y[i]])

# Váha prvního bodu
for i in range(1, 11, 1):
    vaha1[1], vaha2[1] = i / 10, i
    body1, body2 = np.array(bezier_curve_rational(points, vaha1)), np.array(bezier_curve_rational(points, vaha2))
    plotting(body1, "#8ae158", "Vahy bodu 1", ax[0])
    plotting(body2, "#8ae158", "Vahy bodu 1", ax[0])

    if not rozhodnuti and control(body1, objekt):
        print(f"Délka první trajektorie bez dotyku = {fc_delka(body1)}, váhy = {vaha1}")
        plotting(body1, "#f82020", "Vahy bodu 1", ax[0])
    if not rozhodnuti:
        rozhodnuti = control(body1, objekt)


vaha1 = [1, 1, 1, 1]
vaha2 = [1, 1, 1, 1]
vracene = [0, 0]
delka = 0
rozhodnuti = False

# Váha druhého bodu
for i in range(1, 11, 1):
    vaha1[2], vaha2[2] = i / 10, i
    body1, body2 = np.array(bezier_curve_rational(points, vaha1)), np.array(bezier_curve_rational(points, vaha2))
    plotting(body1, "#5287dd", "Vahy bodu 2", ax[1])
    plotting(body2, "#5287dd", "Vahy bodu 2", ax[1])

    if not rozhodnuti and control(body1, objekt):
        print(f"Délka první trajektorie bez dotyku = {fc_delka(body1)}, váhy = {vaha1}")
        plotting(body1, "#f82020", "Vahy bodu 1", ax[1])
    if not rozhodnuti:
        rozhodnuti = control(body1, objekt)


vaha1 = [1, 1, 1, 1]
vaha2 = [1, 1, 1, 1]
vracene = [0, 0]
delka = 0
rozhodnuti = False

# Váha obou bodů
for i in range(1, 11, 1):
    vaha1[1:3], vaha2[1:3] = [i / 10, i / 10], [i, i]
    body1, body2 = np.array(bezier_curve_rational(points, vaha1)), np.array(bezier_curve_rational(points, vaha2))
    plotting(body1, "#e158d0", "Vahy bodu 1 a 2", ax[2])
    plotting(body2, "#e158d0", "Vahy bodu 1 a 2", ax[2])

    if not rozhodnuti and control(body1, objekt):
        print(f"Délka první trajektorie bez dotyku = {fc_delka(body1)}, váhy = {vaha1}")
        plotting(body1, "#f82020", "Vahy bodu 1", ax[2])
    if not rozhodnuti:
        rozhodnuti = control(body1, objekt)


osobni = self_control(points, objekt, value=[1, 0.1, 0.1, 1])

plotting(osobni, "#f82020", "Optimální vyhodnocená", ax[3])

# Úprava subplotů
for i in ax:
    i.grid()
    i.plot(x, y, 'ro')
    i.plot(x, y, 'b')
    i.plot(objekt[:, 0], objekt[:, 1], 'b')

print(f'Délka původní trasy je {fc_delka(np.array(points))}')
np.set_printoptions(precision=1, suppress=True)
mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
plt.savefig("Bezierovy_vahy", dpi=100)
plt.show()

