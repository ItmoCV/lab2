import cv2
import numpy as np

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

for i in range(10):
    image = cv2.imread(start_images[i])
    template = cv2.imread(cut_images[i])
    draw_result(image, template, start_images[i].replace('start', 'rect_cut'))

for j in range(10):
    image = cv2.imread(start_images[j])
    template = cv2.imread(prework_images[j])
    draw_result(image, template, start_images[j].replace('start', 'rect_prework'))
