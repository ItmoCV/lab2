## Теоретическая база

### Прямой поиск одного изображения на другом (template matching)
Прямой поиск одного изображения на другом (template matching) — метод, основанный на нахождении места на изображении, наиболее похожем на шаблон. Этот процесс включает наложение шаблона на исходное изображение и оценку расхождения между ними с использованием различных метрик.

### Поиск ключевых точек эталона на входном изображении (с помощью SIFT)
Метод SIFT (Scale-Invariant Feature Transform) является одним из наиболее эффективных алгоритмов для обнаружения и описания ключевых точек на изображениях. Он широко применяется в задачах компьютерного зрения, таких как распознавание объектов и сопоставление изображений.

Этапы работы алгоритма SIFT:
1. Обнаружение ключевых точек:
    1. Алгоритм начинается с построения пирамиды Гаусса, где изображение последовательно размазывается с использованием гауссовых фильтров различных масштабов. Это позволяет выявить экстремумы (максимумы и минимумы) в пространстве разностей гауссианов, которые служат ключевыми точками.
    2. Ключевые точки выбираются как локальные максимумы в окрестностях, что обеспечивает их устойчивость к шумам и изменениям масштаба.
2. Описание ключевых точек:
    1. Для каждой найденной ключевой точки создается дескриптор, который представляет собой 128-мерный вектор. Этот вектор формируется на основе градиентов изображения в окрестности ключевой точки, разделенной на сектора. Градиенты взвешиваются для уменьшения влияния изменений положения.
    2. Дескрипторы инвариантны к изменениям освещения и масштабированию, что делает их надежными для сопоставления между различными изображениями.
3. Сопоставление ключевых точек:
    1. На этапе сопоставления дескрипторы ключевых точек из эталонного изображения сравниваются с дескрипторами из входного изображения. Для этого используется евклидово расстояние для нахождения наиболее близких соответствий.

## Описание программы

### Прямой поиск одного изображения на другом (template matching)
Поиск реализован в скрипте [template_mathing.py](./template_matching.py)

Поиск наилучшего места осуществляется с помощью функции `template_matching`, на вход в качестве аргументов принимается изображение, в котором надо найти шаблон, и сам шаблон. В качестве метрики расхождения используется MSE.

Описание алгоритма:
1. С помощью циклов происходит проход по картинки окном размера шаблона;
2. Для каждого окна считается ошибка;
3. Находится окно с наименьшей ошибкой и запоминается координаты верхнего левого угла этого окна.
4. Функция возвращает координаты окна с наименьшей ошибкой

Для визуализации найденной области реализована функция `draw_result`. В качестве аргументов она принимает изображение, на котором был осуществлен поиск шаблона, сам шаблон, и путь для сохранения изображения с выделенной областью. Внутри функции вызывается функция `template_matching, и отображается область на изображении.

### Поиск ключевых точек эталона на входном изображении (с помощью SIFT)
Для обнаружения и описания ключевых точек на обоих изображениях – шаблоне и исходном изображении используется метод SIFT. После нахождения ключевых точек производится их сопоставление с использованием алгоритма KNN с тестом Лоу для фильтрации некачественных совпадений.

В случае успешного нахождения достаточного количества совпадающих ключевых точек программа вычисляет положение и размеры bounding box вокруг найденного шаблона на исходном изображении. Если же количество совпадающих точек недостаточно, выводится сообщение об ошибке.

Описание алгоритма:

1. Загрузка изображений (шаблон и исходное изображение).
2. Преобразование изображений в оттенки серого.
3. Обнаружение и описание ключевых точек с использованием SIFT.
4. Сопоставление ключевых точек между двумя изображениями с использованием метода BFMatcher.
5. Фильтрация совпадений с использованием теста Лоу.
6. Определение координат bounding box на основе расположения совпавших ключевых точек.
7. Отображение результата на исходном изображении.
Результатом выполнения программы является изображение с наложенной рамкой вокруг найденного шаблона, а также координаты углов этой рамки.

## Результаты работы
### Прямой поиск одного изображения на другом (template matching)

Изображения шаблоны приведены ниже:
![](./readme_img/temp.png)

Изображения, на которых осуществлялся поиск:
![](./readme_img/orig.png)

Результат работы алгоритма на обрезанных изображениях:
![](./readme_img/result_cut.png)

Результат работы алгоритма на измененных изображениях:
![](./readme_img/result_prework.png)

Время работы программы:
12.0 минут

### Поиск ключевых точек эталона на входном изображении (с помощью SIFT)

Изображения шаблоны приведены ниже:
![](./readme_img/temp.png)

Изображения, на которых осуществлялся поиск:
![](./readme_img/orig.png)

Результат работы алгоритма на обрезанных изображениях:
![](./readme_img/result_cut_sift.png)

Результат работы алгоритма на измененных изображениях:
![](./readme_img/result_prework_sift.png)

Время работы программы:
3.06 секунды

## Выводы по работе
### Template_matching
Метрика IoU работы template_matching по обрезанным изображениям ![image](https://github.com/user-attachments/assets/b986924d-d831-47c0-a544-0df2bf738d9b)

Средняя метрика IoU работы template_matching по обрезанным изображениям составляет 1.0

Метрика IoU работы template_matching по измененным изображениям ![image](https://github.com/user-attachments/assets/717b9459-8baf-4d65-82f7-d17cd57bc7b2)

Средняя метрика IoU работы template_matching по измененным изображениям составляет 0.9

### Поиск ключевых точек
Метрика IoU работы sift.py по обрезанным и измененным изображениям (идентична) ![](./readme_img/chart1.png)

Средняя метрика IoU работы sift.py по составляет 0.34


## Использованные источники
1. Документация OpenCV: https://docs.opencv.org/
2. Документация NumPy: https://numpy.org/doc/stable/
3. Статься на Habr про нахождение объектов на картинках: https://habr.com/ru/companies/joom/articles/445354/
