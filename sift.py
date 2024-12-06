import cv2 as cv
import numpy as np
import os
import pandas as pd
from time import time

start = time()

def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def find_template(template_path, img_path):
    template = cv.imread(template_path)
    image = cv.imread(img_path)

    gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    keypoints_template, descriptors_template = sift.detectAndCompute(gray_template, None)
    keypoints_image, descriptors_image = sift.detectAndCompute(gray_image, None)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors_template, descriptors_image, k=2)

    # Тест Лоу
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    # Проверяем количество совпадений
    if len(good_matches) >= 3:
        # Получаем координаты ключевых точек шаблона
        template_points = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches])

        # Определяем крайние ключ. точки на шаблоне и центр
        min_x = np.min(template_points[:, 0])
        max_x = np.max(template_points[:, 0])
        min_y = np.min(template_points[:, 1])
        max_y = np.max(template_points[:, 1])
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2

        # Находим центральную точку в изображении, на котором будем искать шаблон
        image_points = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches])
        center_image_x = np.mean(image_points[:, 0])
        center_image_y = np.mean(image_points[:, 1])
        min_x_img = np.min(image_points[:, 0])
        max_x_img = np.max(image_points[:, 0])
        min_y_img = np.min(image_points[:, 1])
        max_y_img = np.max(image_points[:, 1])

        # Считаем расстояние от центральной к.т. шаблона до краев шаблона
        template_height, template_width, _ = template.shape
        distance_to_left = distance((center_x, center_y), (0, center_y))
        distance_to_right = distance((center_x, center_y), (template_width, center_y))
        distance_to_top = distance((center_x, center_y), (center_x, 0))
        distance_to_bottom = distance((center_x, center_y), (center_x, template_height))

        # Вычисляем коэффициенты сжатия по осям x и y
        scale_x = (max_x-min_x) / (max_x_img-min_x_img)
        scale_y = (max_y-min_y) / (max_y_img-min_y_img)

        # Вычисляем координаты для рисования bounding box
        top_left_x = int(center_image_x - distance_to_left * scale_x)
        top_left_y = int(center_image_y - distance_to_top * scale_y)
        bottom_right_x = int(center_image_x + distance_to_right * scale_x)
        bottom_right_y = int(center_image_y + distance_to_bottom * scale_y)      

        # Рисуем bounding box на изображении
        image_with_box = image.copy()
        cv.rectangle(image_with_box, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    else:
        print("Недостаточно качественных совпадений для сопоставления шаблона.")

    return image_with_box, top_left_x, top_left_y, bottom_right_x, bottom_right_y


if __name__ == "__main__":
    start_dir = 'images/start/'
    cut_dir = 'images/cut/'
    prework_dir = 'images/prework/'
    cut_res_dir = 'images/SIFT_cut_res/'
    prework_res_dir = 'images/SIFT_prework_res/'
    cut_res_coords = 'metrics/sift/'
    prework_res_coords = 'metrics/sift/'

    os.makedirs(cut_res_dir, exist_ok=True)
    os.makedirs(prework_res_dir, exist_ok=True)
    os.makedirs(cut_res_coords, exist_ok=True)
    os.makedirs(prework_res_coords, exist_ok=True)

    start_images = ['cats.png', 'cbee.png', 'city.png', 'crobot.png',
                    'hello.png', 'house.png', 'map.png', 'most.png', 
                    'planet.png', 'pumpkin.png']
    

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


    for img_name in start_images:
        img_path = os.path.join(start_dir, img_name)
        img = cv.imread(img_path)
        template_cut_path = os.path.join(cut_dir, img_name)
        template_prework_path = os.path.join(prework_dir, img_name)

        result_cut = find_template(template_cut_path, img_path)
        res_img_cut = result_cut[0]
        cv.imwrite(os.path.join(cut_res_dir, img_name), res_img_cut)
        name_cut.append(os.path.splitext(img_name)[0])
        x_min_cut.append(result_cut[1])
        y_min_cut.append(result_cut[2])
        x_max_cut.append(result_cut[3])
        y_max_cut.append(result_cut[4])
        
        result_prework = find_template(template_cut_path, img_path)
        res_img_prework = result_prework[0]
        cv.imwrite(os.path.join(prework_res_dir, img_name), res_img_prework)
        name_prework.append(os.path.splitext(img_name)[0])
        x_min_prework.append(result_cut[1])
        y_min_prework.append(result_cut[2])
        x_max_prework.append(result_cut[3])
        y_max_prework.append(result_cut[4])


    pd.DataFrame({'name': name_cut,
                'x_min': x_min_cut,
                'y_min': y_min_cut,
                'x_max': x_max_cut,
                'y_max': y_max_cut}).to_excel(r"metrics/sift/coord_cut.xlsx")

    pd.DataFrame({'name': name_prework,
                'x_min': x_min_prework,
                'y_min': y_min_prework,
                'x_max': x_max_prework,
                'y_max': y_max_prework}).to_excel(r"metrics/sift/coord_prework.xlsx")
    
    end = time()
    print(f'Время работы программы: {np.round(end - start, 2)} сек.')
