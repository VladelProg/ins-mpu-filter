import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QHBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import serial

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, *rest = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs3d)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, *rest = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

class IMUVisualizer(QMainWindow):
    MAX_POINTS = 100

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Визуализация с Arduino")
        self.setGeometry(100, 100, 1400, 900)

        self.serial_port = None
        self.data_x = []
        self.data_y = []
        self.data_z = []
        self.roll_data = []
        self.pitch_data = []
        self.yaw_data = []

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()

        self.connect_button = QPushButton("Подключиться к COM-порту")
        self.connect_button.clicked.connect(self.toggle_serial)
        left_layout.addWidget(self.connect_button)

        self.label_status = QLabel("Статус: не подключено")
        self.label_drift = QLabel("Дрейф: -- м")
        left_layout.addWidget(self.label_status)
        left_layout.addWidget(self.label_drift)

        layout.addLayout(left_layout)

        # График траектории
        self.fig_traj = Figure(figsize=(8, 6))
        self.canvas_traj = FigureCanvas(self.fig_traj)
        self.ax_traj = self.fig_traj.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas_traj)

        # График ориентации
        self.fig_orient = Figure(figsize=(4, 4))
        self.canvas_orient = FigureCanvas(self.fig_orient)
        self.ax_orient = self.fig_orient.add_subplot(111, projection='3d')
        self.plot_orientation()
        layout.addWidget(self.canvas_orient)

        self.setCentralWidget(main_widget)

    def toggle_serial(self):
        if not self.serial_port:
            try:
                self.serial_port = serial.Serial('COM3', 9600, timeout=1)
                self.connect_button.setText("Отключиться от COM-порта")
                self.label_status.setText("Статус: подключено")
                self.timer.start(50)
            except Exception as e:
                self.label_status.setText(f"Ошибка подключения: {e}")
        else:
            self.serial_port.close()
            self.serial_port = None
            self.connect_button.setText("Подключиться к COM-порту")
            self.label_status.setText("Статус: не подключено")
            self.timer.stop()

    def update_plot(self):
        while self.serial_port and self.serial_port.in_waiting > 0:
            line = self.serial_port.readline().decode().strip()
            print(line)
            if line.startswith("data:"):
                parts = line.split()[1:]  # пропускаем метку

                if len(parts) == 6:
                    x, y, z, roll_deg, pitch_deg, yaw_deg = map(float, parts)

                    self.data_x.append(x)
                    self.data_y.append(y)
                    self.data_z.append(z)
                    self.roll_data.append(roll_deg)
                    self.pitch_data.append(pitch_deg)
                    self.yaw_data.append(yaw_deg)

                    # Ограничиваем размер буфера
                    for data in [self.data_x, self.data_y, self.data_z,
                                 self.roll_data, self.pitch_data, self.yaw_data]:
                        if len(data) > self.MAX_POINTS:
                            data.pop(0)

        self.plot_trajectory()
        self.plot_orientation()

    def plot_trajectory(self):
        if not self.data_x:
            return

        self.ax_traj.clear()
        self.ax_traj.plot(self.data_x, self.data_y, self.data_z)
        self.ax_traj.set_xlabel("X (м)")
        self.ax_traj.set_ylabel("Y (м)")
        self.ax_traj.set_zlabel("Z (м)")
        self.ax_traj.set_title("Траектория (построенная на Arduino)")

        drift = np.linalg.norm([self.data_x[-1], self.data_y[-1], self.data_z[-1]])
        self.label_drift.setText(f"Дрейф: {drift:.3f} м")

        self.canvas_traj.draw()

    def plot_orientation(self):
        from matplotlib.transforms import Affine2D, IdentityTransform
        self.ax_orient.clear()

        # Примерное построение матрицы поворота из углов
        roll = np.radians(self.roll_data[-1]) if self.roll_data else 0
        pitch = np.radians(self.pitch_data[-1]) if self.pitch_data else 0
        yaw = np.radians(self.yaw_data[-1]) if self.yaw_data else 0

        R = euler_to_rot(roll, pitch, yaw)

        origin = [0, 0, 0]
        scale = 1.0

        x_axis = R @ [scale, 0, 0]
        y_axis = R @ [0, scale, 0]
        z_axis = R @ [0, 0, scale]

        self.ax_orient.add_artist(Arrow3D(
            [origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]],
            mutation_scale=10, lw=2, arrowstyle="-|>", color="r"
        ))
        self.ax_orient.add_artist(Arrow3D(
            [origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]],
            mutation_scale=10, lw=2, arrowstyle="-|>", color="g"
        ))
        self.ax_orient.add_artist(Arrow3D(
            [origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]],
            mutation_scale=10, lw=2, arrowstyle="-|>", color="b"
        ))

        self.ax_orient.set_xlim([-1.5, 1.5])
        self.ax_orient.set_ylim([-1.5, 1.5])
        self.ax_orient.set_zlim([-1.5, 1.5])
        self.ax_orient.set_box_aspect([1, 1, 1])
        self.canvas_orient.draw()

def euler_to_rot(r, p, y):
    """Преобразование Эйлера в матрицу поворота"""
    cr, cp, cy = np.cos(r), np.cos(p), np.cos(y)
    sr, sp, sy = np.sin(r), np.sin(p), np.sin(y)

    Rx = np.array([[1, 0, 0],
                  [0, cr, -sr],
                  [0, sr, cr]])

    Ry = np.array([[cp, 0, sp],
                  [0, 1, 0],
                  [-sp, 0, cp]])

    Rz = np.array([[cy, -sy, 0],
                  [sy, cy, 0],
                  [0, 0, 1]])

    return Rz @ Ry @ Rx

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IMUVisualizer()
    window.show()
    sys.exit(app.exec_())
