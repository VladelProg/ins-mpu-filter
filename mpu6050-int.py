import sys
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QLabel, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from madgwick_filter import Madgwick
import serial

# === RK4 функция интегрирования ===
def rk4_step(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    k2 = f(t + dt / 2, y + dt * k1 / 2, *args)
    k3 = f(t + dt / 2, y + dt * k2 / 2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def deriv(t, state, acceleration):
    return np.array([
        state[3], state[4], state[5],
        acceleration[0], acceleration[1], acceleration[2]
    ])

# === Класс стрелок для осей ===
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

# === Главное окно приложения с Serial ===
class RealTimeIMUViewer(QMainWindow):
    MAX_POINTS = 100  # показываем последние N точек
    ALPHA = 0.95       # коэффициент low-pass фильтра

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Реальное время IMU")
        self.setGeometry(100, 100, 1800, 900)

        self.init_ui()
        self.serial_port = None
        self.start_time = None

        # Переменные для фильтрации и интегрирования
        self.mad = Madgwick(sample_period=1/100, beta=0.1)
        self.state = np.zeros(6)  # x, y, z, vx, vy, vz
        self.trajectory = []
        self.rotations = []

        # Углы Эйлера
        self.roll_data = []
        self.pitch_data = []
        self.yaw_data = []
        self.acc_roll_data = []
        self.acc_pitch_data = []

        # Фильтрованные данные
        self.ax_filtered = 0
        self.ay_filtered = 0
        self.az_filtered = 0
        self.gx_filtered = 0
        self.gy_filtered = 0
        self.gz_filtered = 0

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()

        self.connect_button = QPushButton("Подключиться к COM-порту")
        self.connect_button.clicked.connect(self.toggle_serial)
        left_layout.addWidget(self.connect_button)

        self.label_drift = QLabel("Дрейф: -- м")
        self.label_status = QLabel("Статус: не подключено")

        left_layout.addWidget(self.label_status)
        left_layout.addWidget(self.label_drift)

        layout.addLayout(left_layout)

        # График траектории
        self.fig_traj = Figure(figsize=(8, 6))
        self.canvas_traj = FigureCanvas(self.fig_traj)
        self.ax_traj = self.fig_traj.add_subplot(111, projection='3d')
        self.ax_traj.set_title("Траектория в реальном времени")
        layout.addWidget(self.canvas_traj)

        # График ориентации
        self.fig_orient = Figure(figsize=(4, 4))
        self.canvas_orient = FigureCanvas(self.fig_orient)
        self.ax_orient = self.fig_orient.add_subplot(111, projection='3d')
        self.ax_orient.set_title("Ориентация объекта")
        self.draw_orientation(np.eye(3))  # начальная ориентация
        layout.addWidget(self.canvas_orient)

        # График углов Roll/Pitch/Yaw
        self.fig_angles = Figure(figsize=(4, 4))
        self.canvas_angles = FigureCanvas(self.fig_angles)
        self.ax_angles = self.fig_angles.add_subplot(111)
        self.ax_angles.set_title("Углы Эйлера (градусы)")
        self.ax_angles.set_ylabel("Угол (°)")
        self.ax_angles.legend(["Roll", "Pitch", "Yaw"])
        layout.addWidget(self.canvas_angles)

        self.setCentralWidget(main_widget)

        # Таймер для обновления графиков
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

    def toggle_serial(self):
        if not self.serial_port:
            try:
                self.serial_port = serial.Serial('COM3', 9600, timeout=1)
                self.connect_button.setText("Отключиться от COM-порта")
                self.label_status.setText("Статус: подключено")
                self.start_time = time.time()
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
            line = self.serial_port.readline().decode('utf-8').strip()
            if line.startswith("a/g:"):
                parts = line.split()[1:]  # пропускаем "a/g:"
                if len(parts) == 11:
                    ax, ay, az, gx, gy, gz, acc_roll_deg, acc_pitch_deg, _, _, _ = map(float, parts)
                    self.process_data(ax, ay, az, gx, gy, gz, acc_roll_deg, acc_pitch_deg)

        self.plot_trajectory()
        self.plot_orientation()
        self.plot_euler_angles()

    def process_data(self, ax, ay, az, gx, gy, gz, acc_roll_deg, acc_pitch_deg):
        now = time.time()
        if not hasattr(self, 'last_time'):
            self.last_time = now
            return

        dt = now - self.last_time
        self.last_time = now

        # Применяем Low-Pass фильтр
        self.ax_filtered = self.ALPHA * ax + (1 - self.ALPHA) * self.ax_filtered
        self.ay_filtered = self.ALPHA * ay + (1 - self.ALPHA) * self.ay_filtered
        self.az_filtered = self.ALPHA * az + (1 - self.ALPHA) * self.az_filtered
        self.gx_filtered = self.ALPHA * gx + (1 - self.ALPHA) * self.gx_filtered
        self.gy_filtered = self.ALPHA * gy + (1 - self.ALPHA) * self.gy_filtered
        self.gz_filtered = self.ALPHA * gz + (1 - self.ALPHA) * self.gz_filtered

        # Обновляем фильтр Маджвика
        self.mad.sample_period = dt
        self.mad.update([self.gx_filtered, self.gy_filtered, self.gz_filtered],
                        [self.ax_filtered, self.ay_filtered, self.az_filtered])

        R = self.mad.get_rotation_matrix()
        acc_global = R @ [self.ax_filtered, self.ay_filtered, self.az_filtered]

        self.state = rk4_step(deriv, now, self.state, dt, acc_global)
        self.trajectory.append(self.state[:3].copy())
        self.rotations.append(R.copy())

        # Сохраняем углы из Маджвика
        roll_mad, pitch_mad, yaw_mad = self.mad.get_euler_angles()
        self.roll_data.append(roll_mad)
        self.pitch_data.append(pitch_mad)
        self.yaw_data.append(yaw_mad)

        # Сохраняем акселерометр
        self.acc_roll_data.append(np.radians(acc_roll_deg))
        self.acc_pitch_data.append(np.radians(acc_pitch_deg))

        # Ограничиваем количество точек
        for data in [self.roll_data, self.pitch_data, self.yaw_data,
                     self.acc_roll_data, self.acc_pitch_data, self.trajectory, self.rotations]:
            if len(data) > self.MAX_POINTS:
                data.pop(0)

    def plot_trajectory(self):
        if not self.trajectory:
            return

        traj = np.array(self.trajectory)
        self.ax_traj.clear()
        self.ax_traj.plot(traj[:, 0], traj[:, 1], traj[:, 2])
        self.ax_traj.set_xlabel("X (м)")
        self.ax_traj.set_ylabel("Y (м)")
        self.ax_traj.set_zlabel("Z (м)")
        self.ax_traj.set_title("Траектория в реальном времени")
        self.canvas_traj.draw()

        drift = np.linalg.norm(self.trajectory[-1]) if self.trajectory else 0
        self.label_drift.setText(f"Дрейф: {drift:.3f} м")

    def plot_orientation(self):
        if not self.rotations:
            return

        R = self.rotations[-1]
        self.ax_orient.clear()
        self.draw_orientation(R)
        self.canvas_orient.draw()

    def plot_euler_angles(self):
        if not self.roll_data or not self.pitch_data or not self.yaw_data:
            return

        ax = self.ax_angles
        ax.clear()
        ax.set_title("Roll / Pitch / Yaw (градусы)")
        ax.set_xlabel("Время (шаги)")
        ax.set_ylabel("Угол (°)")

        times = np.arange(len(self.roll_data))
        ax.plot(times, np.degrees(self.roll_data), 'r-', label="Roll (Madgwick)")
        ax.plot(times, np.degrees(self.acc_roll_data), 'r--', label="Accel Roll")

        ax.plot(times, np.degrees(self.pitch_data), 'g-', label="Pitch (Madgwick)")
        ax.plot(times, np.degrees(self.acc_pitch_data), 'g--', label="Accel Pitch")

        ax.plot(times, np.degrees(self.yaw_data), 'b-', label="Yaw (Madgwick)")

        ax.legend()
        ax.grid(True)
        self.canvas_angles.draw()

    def draw_orientation(self, rotation_matrix=np.eye(3)):
        origin = [0, 0, 0]
        scale = 1.0

        x_axis = rotation_matrix @ [scale, 0, 0]
        y_axis = rotation_matrix @ [0, scale, 0]
        z_axis = rotation_matrix @ [0, 0, scale]

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealTimeIMUViewer()
    window.show()
    sys.exit(app.exec_())
