import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# Время обновления
dt = 0.01
frame_count = 0

# Инициализация состояния
velocity = np.zeros(3)
position = np.zeros(3)
angle = np.zeros(3)  # roll, pitch, yaw
trajectory = []
last_accel = None

# Плоский объект ("дрон")
wing = np.array([
    [-1.5, -0.3, 0],
    [+1.5, -0.3, 0],
    [+1.5, +0.3, 0],
    [-1.5, +0.3, 0],
    [-1.5, -0.3, 0.1],
    [+1.5, -0.3, 0.1],
    [+1.5, +0.3, 0.1],
    [-1.5, +0.3, 0.1]
])

def get_faces(points):
    return [
        [points[j] for j in [0, 1, 2, 3]],
        [points[j] for j in [4, 5, 6, 7]],
        [points[j] for j in [0, 1, 5, 4]],
        [points[j] for j in [2, 3, 7, 6]],
        [points[j] for j in [1, 2, 6, 5]],
        [points[j] for j in [4, 7, 3, 0]]
    ]

def rotate(points, angles):
    roll, pitch, yaw = angles
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return np.dot(points, Rx @ Ry @ Rz.T)

def generate_scenario(frame):
    phase = (frame // 300) % 4
    if phase == 0:
        # Прямой полет
        accel = np.array([0.05, 0.0, 0.0])
        gyro = np.array([0.0, 0.0, 0.0])
    elif phase == 1:
        # Резкий взлет
        accel = np.array([0.05, 0.0, 0.8])
        gyro = np.array([0.0, 0.0, 0.0])
    elif phase == 2:
        # Поворот (yaw)
        accel = np.array([0.05, 0.0, 0.0])
        gyro = np.array([0.0, 0.0, 0.5])
    else:
        # Крен (roll) + небольшой pitch
        accel = np.array([0.05, 0.0, 0.0])
        gyro = np.array([0.2, 0.1, 0.0])
    return accel + np.random.normal(0, 0.01, 3), gyro + np.random.normal(0, 0.01, 3)

# График
fig = plt.figure(figsize=(12, 6))
ax2d = fig.add_subplot(1, 2, 1)
ax3d = fig.add_subplot(1, 2, 2, projection='3d')
line2d, = ax2d.plot([], [], lw=2)
ax2d.set_title("2D траектория")
ax2d.set_xlabel("X")
ax2d.set_ylabel("Y")
ax2d.set_aspect("equal")
ax2d.grid()

ax3d.set_xlim([-2, 2])
ax3d.set_ylim([-2, 2])
ax3d.set_zlim([-1, 2])
ax3d.set_title("Ориентация объекта")
poly = None

def init():
    ax2d.set_xlim(-1, 1)
    ax2d.set_ylim(-1, 1)
    return line2d,

def update(frame):
    global velocity, position, last_accel, angle, poly, frame_count
    frame_count += 1

    accel, gyro = generate_scenario(frame_count)

    if last_accel is not None:
        velocity += 0.5 * (accel + last_accel) * dt
        position += velocity * dt
    last_accel = accel
    angle += gyro * dt

    # 2D график
    trajectory.append(position.copy())
    data = np.array(trajectory)
    line2d.set_data(data[:, 0], data[:, 1])
    ax2d.relim()
    ax2d.autoscale_view()

    # 3D объект
    rotated = rotate(wing, angle)
    if poly:
        poly.remove()
    poly = Poly3DCollection(get_faces(rotated), facecolors='lightgreen', edgecolors='k', alpha=0.8)
    ax3d.add_collection3d(poly)

    return line2d, poly

ani = FuncAnimation(fig, update, init_func=init, interval=10, blit=False)
plt.tight_layout()
plt.show()
