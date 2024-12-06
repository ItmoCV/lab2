import cv2
import numpy as np
import pandas as pd
import time


start_time = time.time()


def change_images(initial_image, path):
    gaus_image = cv2.GaussianBlur(initial_image, (7, 7), 0)
    cv2.imwrite(path, gaus_image)


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def template_matching(image, template):
    tH, tW = template.shape[:2]

    iH, iW = image.shape[:2]

    best_mse = 10 ** 9
    best_position = (0, 0)

    for y in range(iH - tH + 1):
        for x in range(iW - tW + 1):
            window = image[y:y + tH, x:x + tW]

            error = mse(window, template)

            if error < best_mse:
                best_mse = error
                best_position = (x, y)

    return best_position


def draw_result(image, template, path):
    position = template_matching(image, template)

    if position:
        x, y = position
        im = cv2.rectangle(image, position, (x + template.shape[1], y + template.shape[0]), (0, 255, 0), 2)
        cv2.imwrite(path, im)

        x_min, y_min, x_max, y_max = x, y, x + template.shape[1], y + template.shape[0]
        if 'rect_cut' in path:
            name = path[15:].split('.')[0][1:]
        else:
            name = path[19:].split('.')[0][1:]

        return [x_min, y_min, x_max, y_max, name]
    else:
        print("Шаблон не найден.")


start_images = ["images\start\cats.png", "images\start\cbee.png", "images\start\city.png", "images\start\hello.png",
                  "images\start\planet.png", "images\start\house.png", "images\start\most.png", "images\start\pumpkin.png",
                  "images\start\crobot.png", "images\start\map.png"]

cut_images = ["images\cut\cats.png", "images\cut\cbee.png", "images\cut\city.png", "images\cut\hello.png",
                  "images\cut\planet.png", "images\cut\house.png", "images\cut\most.png", "images\cut\pumpkin.png",
                  "images\cut\crobot.png", "images\cut\map.png"]

prework_images = ["images\prework\cats.png", "images\prework\cbee.png", "images\prework\city.png", "images\prework\hello.png",
                  "images\prework\planet.png", "images\prework\house.png", "images\prework\most.png", "images\prework\pumpkin.png",
                  "images\prework\crobot.png", "images\prework\map.png"]

coord_cut = []
for i in range(10):
    image = cv2.imread(start_images[i])
    template = cv2.imread(cut_images[i])
    coord_cut.append(draw_result(image, template, start_images[i].replace('start', 'rect_cut')))

coord_prework = []
for j in range(10):
    image = cv2.imread(start_images[j])
    template = cv2.imread(prework_images[j])
    coord_prework.append(draw_result(image, template, start_images[j].replace('start', 'rect_prework')))

x_min_cut = []
y_min_cut = []
x_max_cut = []
y_max_cut = []
name_cut = []

x_min_prework = []
y_min_prework = []
x_max_prework = []
y_max_prework = []
name_prework = []

for coord in coord_cut:
    x_min_cut.append(coord[0])
    y_min_cut.append(coord[1])
    x_max_cut.append(coord[2])
    y_max_cut.append(coord[3])
    name_cut.append(coord[4])

for coord in coord_prework:
    x_min_prework.append(coord[0])
    y_min_prework.append(coord[1])
    x_max_prework.append(coord[2])
    y_max_prework.append(coord[3])
    name_prework.append(coord[4])


pd.DataFrame({'name': name_cut,
              'x_min': x_min_cut,
              'y_min': y_min_cut,
              'x_max': x_max_cut,
              'y_max': y_max_cut}).to_excel(r"metrics\template_matching\coord_cut.xlsx")

pd.DataFrame({'name': name_prework,
              'x_min': x_min_prework,
              'y_min': y_min_prework,
              'x_max': x_max_prework,
              'y_max': y_max_prework}).to_excel(r"metrics\template_matching\coord_prework.xlsx")

end_time = time.time()
execution_time = end_time - start_time

print(f"Время выполнения программы: {execution_time // 60} минут, {execution_time % 60} секунд")
