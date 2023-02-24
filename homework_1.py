import matplotlib.pyplot as plt
import numpy as np
import cv2

size = 8
plt.rcParams["figure.figsize"] = (size, size)


# Parsing data from txt file
def parse(notepad):
    x, y, a = [], [], []
    rays = []

    for el in notepad:
        temp = el.split('; ')

        pose = temp[0]
        pose = pose.split(', ')
        pose = list(map(float, pose))
        x.append(pose[0])
        y.append(pose[1])
        a.append(pose[2])

        temp = temp[1]
        temp = temp.split(', ')
        rays.append(list(map(float, temp)))

    return x, y, a, rays


# Transforming vectors from polar coordinates to cartesian coordinates
def get_obstacles(x, y, a, rays):
    x_obs = []
    y_obs = []
    a_rays = [-120 + 240 / (len(rays[0]) - 1) * i for i in range(len(rays[0]))]

    for i in range(len(x)):
        for j in range(len(rays[i])):
            if 1 < rays[i][j] < 5.6:
                phi = a[i] - np.radians(a_rays[j])
                x_obs.append(x[i] + 0.3 * np.cos(a[i]) + rays[i][j] * np.cos(phi))
                y_obs.append(y[i] + 0.3 * np.sin(a[i]) + rays[i][j] * np.sin(phi))

    return x_obs, y_obs


txt = open(r"D:\_Institute\my_files\examp13.txt")
x, y, a, rays = parse(txt)
x_obs, y_obs = get_obstacles(x, y, a, rays)


# Original image
figure = plt.figure()
plt.axis('off')
plt.plot(x_obs, y_obs, "r.")
plt.xlim(min(x_obs), max(x_obs))
plt.ylim(min(y_obs), max(y_obs))


# Transforming obstacles from plt format to cv2 format
figure.canvas.draw()
image_from_plot = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
im = cv2.cvtColor(image_from_plot.reshape(figure.canvas.get_width_height()[::-1] + (3,)), cv2.COLOR_RGBA2BGR)


# Transforming trajectory from plt format to cv2 format
figure = plt.figure()
plt.axis('off')
plt.xlim(min(x_obs), max(x_obs))
plt.ylim(min(y_obs), max(y_obs))
plt.plot(x, y, 'r')
plt.plot(x, y, 'r.')

figure.canvas.draw()
image_from_plot = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
trajectory = cv2.cvtColor(image_from_plot.reshape(figure.canvas.get_width_height()[::-1] + (3,)),  cv2.COLOR_RGBA2BGR)
trajectory = cv2.bitwise_not(trajectory)


# Using Canny
im_edge = cv2.Canny(im, 50, 50)
cv2.imshow('Canny', im_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Using Ramer-Douglas-Peucker algorithm
contours, hierarchy = cv2.findContours(im_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
t = np.zeros(im.shape)

for cnt in contours:
    if len(cnt) > 80:
        epsilon = 1/900 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(t, [approx], 0, (0, 255, 0), 1)

im_final = t + trajectory
cv2.imshow('Approx contours + trajectory', im_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
