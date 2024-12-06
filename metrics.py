import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


coord = pd.read_excel('metrics/coord.xlsx')
coord_cut = pd.read_excel('metrics/template_matching/coord_cut.xlsx')
coord_prework = pd.read_excel('metrics/template_matching/coord_prework.xlsx')


def calculate_iou(boxA, boxB):
    # Распаковка координат боксов
    xA, yA, xB, yB = boxA
    xC, yC, xD, yD = boxB

    # Вычисление координат пересечения
    xI = max(xA, xC)
    yI = max(yA, yC)
    xU = min(xB, xD)
    yU = min(yB, yD)

    # Вычисление площади пересечения
    intersection_width = max(0, xU - xI)
    intersection_height = max(0, yU - yI)
    intersection_area = intersection_width * intersection_height

    # Вычисление площади обоих боксов
    boxA_area = (xB - xA) * (yB - yA)
    boxB_area = (xD - xC) * (yD - yC)

    # Вычисление площади объединения
    union_area = boxA_area + boxB_area - intersection_area

    # Вычисление IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


metric_cut = dict()

for i, row in coord.iterrows():
    box_A = float(row['x_min']), float(row['y_min']), float(row['x_max']), float(row['y_max'])
    box_B = float(coord_cut['x_min'][i]), float(coord_cut['y_min'][i]), float(coord_cut['x_max'][i]), float(
        coord_cut['y_max'][i])

    metric_cut[row['name']] = calculate_iou(box_A, box_B)

print(f'Средняя метрика для обработанных изображений {np.mean(list(metric_cut.values()))}')

metric_prework = dict()

for i, row in coord.iterrows():
    box_A = float(row['x_min']), float(row['y_min']), float(row['x_max']), float(row['y_max'])
    box_B = float(coord_prework['x_min'][i]), float(coord_prework['y_min'][i]), float(coord_prework['x_max'][i]), float(
        coord_prework['y_max'][i])

    metric_prework[row['name']] = calculate_iou(box_A, box_B)

print(f'Средняя метрика для обработанных изображений {np.mean(list(metric_prework.values()))}')

x = list(metric_prework.keys())
y = list(metric_prework.values())

plt.bar(height=y, x=x)
plt.show()

x = list(metric_cut.keys())
y = list(metric_cut.values())

plt.bar(height=y, x=x)
plt.show()
