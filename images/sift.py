import cv2
import numpy as np
import os

def find_object(img, template):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    
    # Ищем ключевые точки и дескрипторы
    kp_template, des_template = sift.detectAndCompute(template_gray, None)
    kp_img, des_img = sift.detectAndCompute(img_gray, None)
    
    # FLANN
    index_params = dict(algorithm=0, trees=5) # KDTree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_template, des_img, k=2)
    
    # Тест Лоу
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            
    # Если достаточно совпадающих точек, считаем гомографию
    min_match_count = 5
    if len(good_matches) > min_match_count:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        
        h, w = template_gray.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        # bounding box
        img = cv2.polylines(img, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    return img


if __name__ == "__main__":
    start_dir = 'images/start/'
    cut_dir = 'images/cut/'
    prework_dir = 'images/prework/'
    cut_res_dir = 'images/SIFT_cut_res/'
    prework_res_dir = 'images/SIFT_prework_res/'

    os.makedirs(cut_res_dir, exist_ok=True)
    os.makedirs(prework_res_dir, exist_ok=True)

    start_images = ['cats.png', 'cbee.png', 'city.png', 'crobot.png',
                    'hello.png', 'house.png', 'map.png', 'most.png', 
                    'planet.png', 'pumpkin.png']

    for img_name in start_images:
        img_path = os.path.join(start_dir, img_name)
        img = cv2.imread(img_path)

        template_cut_path = os.path.join(cut_dir, img_name)
        template_cut = cv2.imread(template_cut_path)
        result_cut = find_object(img, template_cut)
        cv2.imwrite(os.path.join(cut_res_dir, img_name), result_cut)

        template_prework_path = os.path.join(prework_dir, img_name)
        template_prework = cv2.imread(template_prework_path)
        result_prework = find_object(img, template_prework)
        cv2.imwrite(os.path.join(prework_res_dir, img_name), result_prework)