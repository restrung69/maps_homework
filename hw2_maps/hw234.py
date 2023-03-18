from tkinter import *
from PIL import ImageTk, Image

import numpy as np
import cv2
from heapq import *

import random
import math

size_img = 900  # в пикселях
grid_cells = 100  # по умолчанию


def draw_tiles(array, cells):
    size = max(array.shape)
    cells = cells if cells < size else size
    interval = round(size / cells)
    # Коэффициент для интерполяции
    fill = min(1.156e-4 * cells ** 3 - 3.855e-2 * cells ** 2 + 3.567 * cells - 46.896, 0.5)
    new_array = np.zeros((size, size))

    for i in range(0, size - interval, interval):
        for j in range(0, size - interval, interval):
            if len(array[i:i + interval, j:j + interval].nonzero()[0]) > fill * interval ** 2:
                new_array[i:i + interval, j:j + interval] = 1.0

    return new_array


def draw_grid(array, cells):
    size = max(array.shape)
    cells = cells if cells < size else size

    for i in range(1, cells):
        interval = round(i * size / cells)
        cv2.line(array, (0, interval), (size, interval), color=(0.5, 0.5, 0.5))
        cv2.line(array, (interval, 0), (interval, size), color=(0.5, 0.5, 0.5))

    return array


def check_nodes(x, y, size):
    check_next_node = lambda x, y: True if 0 <= x < size and 0 <= y < size else False
    moves = [-1, 0], [0, -1], [1, 0], [0, 1]
    return [(x + dx, y + dy) for dx, dy in moves if check_next_node(x + dx, y + dy)]


def make_grid(array, cells):
    size = max(array.shape)
    cells = cells if cells < size else size
    interval = round(size / cells)
    fill = min(1.156e-4 * cells ** 3 - 3.855e-2 * cells ** 2 + 3.567 * cells - 46.896, 0.5)
    gridmap = np.zeros((cells, cells))

    for i in range(0, size - interval, interval):
        for j in range(0, size - interval, interval):
            if len(array[i:i + interval, j:j + interval].nonzero()[0]) > fill * interval ** 2:
                gridmap[i // interval, j // interval] = 1.0

    return gridmap


def make_graph(array, cells):
    gridmap = make_grid(array, cells) if cells <= size_img / 5 else array

    nodes = {}
    edges = {}
    for x in range(cells):
        for y in range(cells):
            color = gridmap[x][y]
            if color == 0:  # черный пиксель является узлом
                nodes[(x, y)] = []
                neighbors = check_nodes(x, y, cells)

                for neigh in neighbors:
                    nx, ny = neigh
                    neighbor_color = gridmap[nx][ny]
                    if neighbor_color == 0:  # соседняя клетка является узлом
                        distance = 1  # расстояние между клетками
                        edges.setdefault((x, y), []).append(((nx, ny), distance))

    return nodes, edges


def draw_path(gridmap, path, visited, cells):
    size = max(gridmap.shape)
    cells = cells if cells < size else size
    scale = round(size / cells)

    for i, j in visited:
        gridmap[i * scale:i * scale + scale, j * scale:j * scale + scale, :] = np.array([0.2, 0.2, 0])

    for i, j in path:
        gridmap[i * scale:i * scale + scale, j * scale:j * scale + scale, :] = np.array([1.0, 0, 0])

    # Отметить начало и конец
    gridmap[path[0][0] * scale:(path[0][0] + 1) * scale, path[0][1] * scale:(path[0][1] + 1) * scale, :] = np.array(
        [1.0, 0, 1.0])
    gridmap[path[-1][0] * scale:(path[-1][0] + 1) * scale, path[-1][1] * scale:(path[-1][1] + 1) * scale, :] = np.array(
        [1.0, 0, 1.0])

    return gridmap


""" Дейкстра """


def Dijkstra(graph, start, destination):
    if graph.get(start) is None or graph.get(destination) is None:
        return [destination, start], [], "No path"

    queue = []
    heappush(queue, start)
    distance = {start: 0}
    visited = {start: None}

    # Сложность E * log(V)
    # Оптимизация через кучу
    while queue:
        cur_node = heappop(queue)

        #         # Алгоритм Дейкстры для оптимального решения
        #         # требует просмотреть весь граф
        #         if cur_node == destination:
        #             break

        next_nodes = graph.get(cur_node)
        if next_nodes is not None:
            for next_node in next_nodes:
                neigh_node, neigh_dist = next_node
                new_dist = distance[cur_node] + neigh_dist

                if neigh_node not in distance or new_dist < distance[neigh_node]:
                    heappush(queue, neigh_node)
                    distance[neigh_node] = new_dist
                    visited[neigh_node] = cur_node

        else:
            return [destination, start], [], "No path"

    path = [destination]
    while path[-1] != start:
        tmp = visited.get(path[-1])
        path.append(tmp)

    return path, visited, distance[destination]


""" A* """


# Функция оценки расстояния для А* и RRT
def manhattan(start, destination):
    return abs(start[0] - destination[0]) + abs(start[1] - destination[1])


# def euclid(start, destination):
#     return ((start[0] - destination[0]) ** 2 + (start[1] - destination[1]) ** 2) ** 0.5


def A_star(graph, start, destination):
    if graph.get(start) is None or graph.get(destination) is None:
        return [destination, start], [], "No path"

    queue = []
    heappush(queue, (0, start))
    distance = {start: 0}
    visited = {start: None}

    while queue:
        cur_dist, cur_node = heappop(queue)

        if cur_node == destination:
            break

        next_nodes = graph.get(cur_node)
        if next_nodes is not None:
            for next_node in next_nodes:
                neigh_node, neigh_dist = next_node
                new_dist = distance[cur_node] + neigh_dist

                # Добавляется приоритет клеток с помощью эвристической функции
                if neigh_node not in distance or new_dist < distance[neigh_node]:
                    priority = new_dist + manhattan(neigh_node, destination)
                    # priority = new_dist + euclid(neigh_node, destination)
                    heappush(queue, (priority, neigh_node))
                    distance[neigh_node] = new_dist
                    visited[neigh_node] = cur_node

        else:
            return [destination, start], [], "No path"

    path = [destination]
    while path[-1] != start:
        tmp = visited.get(path[-1])
        path.append(tmp)

    return path, visited, distance[destination]


""" RRT """
RADIUS = int(size_img // 100)       # радиус окрестности для поиска ближайшей точки
MAX_ITERATIONS = 10000              # максимальное число итераций для построения дерева
STEP_SIZE = int(size_img // 100)    # шаг для добавления новых точек


def is_obstacle(array, new_point):
    return True if array[new_point] else False


def check_new_point(new_point):
    y, x = new_point
    check = True if 0 <= x < size_img and 0 <= y < size_img else False
    return check


def find_nearest(tree, point):
    # Найти ближайшую точку в дереве
    nearest_point = tree[0]
    nearest_distance = manhattan(nearest_point, point)
    d = 1e7  # ближайшее расстояние в return
    for p in tree:
        d = manhattan(p, point)
        if d < nearest_distance:
            nearest_distance = d
            nearest_point = p
    return nearest_point, d


def generate_random_point(size=size_img):
    # Выбрать случайную точку на картинке
    x = random.randint(0, size_img - 1)
    y = random.randint(0, size_img - 1)
    return (x, y)


def add_point(array, tree, new_point, nearest_point):
    # Добавить новую точку к дереву
    if manhattan(nearest_point, new_point) > STEP_SIZE:
        theta = math.atan2(new_point[1] - nearest_point[1], new_point[0] - nearest_point[0])
        new_point = (
        round(nearest_point[0] + STEP_SIZE * math.cos(theta)), round(nearest_point[1] + STEP_SIZE * math.sin(theta)))

    if check_new_point(new_point) and not is_obstacle(array, new_point):
        return new_point
    return None


def build_RRT(array, start, goal):
    # Построить дерево от начальной точки до целевой
    tree = [start]
    parents = {start: None}
    distance = 0

    for i in range(MAX_ITERATIONS):
        rand_point = generate_random_point()
        nearest_point, d = find_nearest(tree, rand_point)
        new_point = add_point(array, tree, rand_point, nearest_point)
        if new_point is None or new_point == nearest_point:
            continue
        tree.append(new_point)
        parents[new_point] = nearest_point

        if manhattan(new_point, goal) < RADIUS:
            add_point(array, tree, goal, new_point)
            parents[goal] = new_point
            return tree, parents
    return None, None


def plot_RRT(map_img, tree, parents, start, goal):
    """Отображает дерево RRT и путь к целевой точке"""
    array = cv2.merge([map_img, map_img, map_img])
    if tree is None:
        return
    for p in tree:
        if parents.get(p) is not None:
            cv2.line(array, (p[1], p[0]), (parents[p][1], parents[p][0]), color=(255 // 2, 255 // 2, 0), thickness=2)

    path = [goal]
    distance = 0
    while path[-1] != start:
        distance += manhattan(path[-1], parents[path[-1]])
        path.append(parents[path[-1]])

    for i in range(len(path) - 1):
        cv2.line(array, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), color=(0, 255, 0), thickness=2)

    cv2.circle(array, (start[1], start[0]), RADIUS, color=(0, 0, 255), thickness=1)
    cv2.circle(array, (goal[1], goal[0]), RADIUS, color=(0, 0, 255), thickness=1)

    return array, distance


img_path = r"hw_map.png"
interp = cv2.INTER_NEAREST  # cv2.INTER_AREA
map_original = cv2.imread(img_path)
map_original = cv2.resize(map_original, (size_img, size_img), interpolation=interp)

offset_x = 50

img = None
img = cv2.cvtColor(map_original, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)

start_pos = None
destination_pos = None
state = None

root = Tk()

root.title("Path planning algorithms")
root.geometry("1200x900")
root["bg"] = "#C0C0C0"
root.resizable(width=False, height=False)


def change_cells(cells):
    global grid_cells
    grid_cells = cells


def get_mouse_pos(event):
    return event.x, event.y


def get_start(event):
    x, y = get_mouse_pos(event)
    global start_pos
    start_pos = (y, x)
    string = ', '.join([str(x), str(y)])
    label_start["text"] = f"Start: {string}"


def get_destination(event):
    x, y = get_mouse_pos(event)
    global destination_pos
    destination_pos = (y, x)
    string = ', '.join([str(x), str(y)])
    label_end["text"] = f"Destination: {string}"


def get_distance(state, distance):
    if type(distance) != str:
        if state == "RRT":
            label_distance["text"] = f"Distance {state} = {distance} px"
        else:
            label_distance["text"] = f"Distance {state} = {distance * size_img // grid_cells} px"
    else:
        label_distance["text"] = f"No path found"


def change_map(state):
    if start_pos is None or destination_pos is None or state is None:
        return

    IMG_LOCAL = None
    # RRT
    if state == "RRT":
        map_img = np.amax(map_original, axis=-1)
        _, map_img = cv2.threshold(map_img, 200, 255, cv2.THRESH_BINARY)
        tree, parents = build_RRT(map_img, start_pos, destination_pos)

        if tree is not None:
            IMG_LOCAL, distance = plot_RRT(map_img, tree, parents, start_pos, destination_pos)
            get_distance(state, distance)
        else:
            IMG_LOCAL = map_img
            cv2.circle(IMG_LOCAL, (start_pos[1], start_pos[0]), RADIUS, color=(0, 0, 255), thickness=1)
            cv2.circle(IMG_LOCAL, (destination_pos[1], destination_pos[0]), RADIUS, color=(0, 0, 255), thickness=1)
            get_distance(state, '')

    elif grid_cells <= size_img / 5:
        map_img = np.amax(map_original, axis=-1)
        _, map_img = cv2.threshold(map_img, 200, 255, cv2.THRESH_BINARY)
        map_img = draw_tiles(map_img, grid_cells)
        scale = size_img / grid_cells
        start = (round(start_pos[0] / scale), round(start_pos[1] / scale))  # --(527, 304)
        destination = (round(destination_pos[0] / scale), round(destination_pos[1] / scale))  # --(70, 42)

        nodes, edges = make_graph(map_img, grid_cells)

        # Dijkstra
        if state == "Dijkstra":
            path, visited, distance = Dijkstra(edges, start, destination)
            get_distance(state, distance)

        # A*
        elif state == "A*":
            path, visited, distance = A_star(edges, start, destination)
            get_distance(state, distance)

        MAP_WITH_PATH = cv2.merge([map_img, map_img, map_img])
        MAP_WITH_PATH = draw_path(MAP_WITH_PATH, path, visited, grid_cells)
        MAP_WITH_CELLS = draw_grid(MAP_WITH_PATH, grid_cells)
        IMG_LOCAL = (MAP_WITH_CELLS * 255).astype(np.uint8)

    else:
        map_img = np.amax(map_original, axis=-1)
        _, map_img = cv2.threshold(map_img, 200, 255, cv2.THRESH_BINARY)

        start = (start_pos[0], start_pos[1])
        destination = (destination_pos[0], destination_pos[1])

        nodes, edges = make_graph(map_img, grid_cells)

        # Dijkstra
        if state == "Dijkstra":
            path, visited, distance = Dijkstra(edges, start, destination)
            get_distance(state, distance)

        # A*
        elif state == "A*":
            path, visited, distance = A_star(edges, start, destination)
            get_distance(state, distance)

        MAP_WITH_PATH = cv2.merge([map_img, map_img, map_img])

        for i, j in visited:
            MAP_WITH_PATH[i, j, :] = np.array([255 // 5, 255 // 5, 0])

        for i, j in path:
            MAP_WITH_PATH[i, j, :] = np.array([0, 0, 255])

        MAP_WITH_PATH[path[0][0], path[0][1], :] = np.array([255, 255, 0])
        MAP_WITH_PATH[path[-1][0], path[-1][1], :] = np.array([255, 255, 0])

        IMG_LOCAL = MAP_WITH_PATH

    IMG_LOCAL = cv2.cvtColor(IMG_LOCAL, cv2.COLOR_BGR2RGB)
    IMG_LOCAL = Image.fromarray(IMG_LOCAL)
    IMG_LOCAL = ImageTk.PhotoImage(IMG_LOCAL)
    label_map.configure(image=IMG_LOCAL)
    label_map.image = IMG_LOCAL


img = ImageTk.PhotoImage(img)


rrt_img = PhotoImage(file="rrt.png")
astar_img = PhotoImage(file="astar.png")
dijkstra_img = PhotoImage(file="dijkstra.png")
cell_size_img = PhotoImage(file="cell_size.png")
img_50x50 = PhotoImage(file="50x50.png")
img_100x100 = PhotoImage(file="100x100.png")
img_150x150 = PhotoImage(file="150x150.png")
img_pix_cell = PhotoImage(file="1px1cell.png")


label_map = Label(root, image=img)
label_map.place(x=0, y=0)


label_start = Label(root, text="Choose start with LMB", width=25, height=2, font=14, bg="#C0C0C0")
label_start.place(x=920, y=20)

label_end = Label(root, text="Choose destination with RMB", width=25, height=2, font=14, bg="#C0C0C0")
label_end.place(x=920, y=60)

label_distance = Label(root, text="No path found", width=25, height=2, font=14, bg="#C0C0C0")
label_distance.place(x=920, y=100)

algorithm_label = Label(root, text="Choose algorithm:", width=20, height=2, font=14, bg="#C0C0C0")
algorithm_label.place(x=920, y=140)


Button(root, command=lambda: change_map("Dijkstra"), image=dijkstra_img, borderwidth=0).place(x=920, y=185)
Button(root, command=lambda: change_map("A*"), image=astar_img, borderwidth=0).place(x=920, y=265)
Button(root, command=lambda: change_map("RRT"), image=rrt_img, borderwidth=0).place(x=920, y=345)


Label(root, image=cell_size_img, borderwidth=0).place(x=920, y=500)


Button(root, command=lambda: change_cells(50), image=img_50x50, borderwidth=0).place(x=920, y=540)
Button(root, command=lambda: change_cells(100), image=img_100x100, borderwidth=0).place(x=920, y=580)
Button(root, command=lambda: change_cells(150), image=img_150x150, borderwidth=0).place(x=920, y=620)
Button(root, command=lambda: change_cells(900), image=img_pix_cell, borderwidth=0).place(x=920, y=660)


label_map.bind("<Button-1>", get_start)
label_map.bind("<Button-3>", get_destination)

root.mainloop()

