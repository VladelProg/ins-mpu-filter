import sys
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QLabel, QHBoxLayout, QFileDialog, QSlider
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from madgwick_filter import Madgwick
import serial
import pandas as pd

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

# === Главный класс приложения с Serial и CSV ===
class IMUApp(QMainWindow):
    MAX_POINTS = 100
    ALPHA = 0.95

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Траектория: Serial + CSV")
        self.setGeometry(100, 100, 1800, 900)

        # Переменные для реального времени
        self.serial_port = None
        self.start_time = None
        self.mad_realtime = Madgwick(sample_period=1/100, beta=0.1)
        self.state_realtime = np.zeros(6)
        self.trajectory_realtime = []
        self.rotations_realtime = []

        # Углы из Serial
        self.roll_data = []
        self.pitch_data = []
        self.yaw_data = []
        self.acc_roll_data = []
        self.acc_pitch_data = []

        # Для CSV
        self.csv_times = []
        self.csv_accs = []
        self.csv_gyros = []
        self.csv_traj = []
        self.csv_rotations = []

        # Углы из CSV
        self.roll_data_csv = []
        self.pitch_data_csv = []
        self.yaw_data_csv = []

        # Слайдер
        self.slider_value = 0

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()

        self.btn_connect = QPushButton("Подключиться к COM-порту")
        self.btn_connect.clicked.connect(self.toggle_serial)
        left_layout.addWidget(self.btn_connect)

        self.btn_load_csv = QPushButton("Загрузить CSV файл")
        self.btn_load_csv.clicked.connect(self.load_csv)
        left_layout.addWidget(self.btn_load_csv)

        self.label_status = QLabel("Статус: не подключено")
        self.label_drift_realtime = QLabel("Дрейф (реальное время): -- м")
        self.label_drift_csv = QLabel("Дрейф (из CSV): -- м")

        left_layout.addWidget(self.label_status)
        left_layout.addWidget(self.label_drift_realtime)
        left_layout.addWidget(self.label_drift_csv)

        # Слайдер прокрутки CSV
        self.slider_label = QLabel("Кадр (CSV): 0")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_change)

        left_layout.addWidget(self.slider_label)
        left_layout.addWidget(self.slider)

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
        self.draw_orientation(np.eye(3))
        layout.addWidget(self.canvas_orient)

        # График углов
        self.fig_angles = Figure(figsize=(4, 4))
        self.canvas_angles = FigureCanvas(self.fig_angles)
        self.ax_angles = self.fig_angles.add_subplot(111)
        self.ax_angles.set_title("Roll / Pitch / Yaw (градусы)")
        self.ax_angles.legend(["Roll", "Pitch", "Yaw"])
        layout.addWidget(self.canvas_angles)

        self.setCentralWidget(main_widget)

    def toggle_serial(self):
        if not self.serial_port:
            try:
                self.serial_port = serial.Serial('COM3', 9600, timeout=1)
                self.btn_connect.setText("Отключиться от COM-порта")
                self.label_status.setText("Статус: подключено")
                self.start_time = time.time()
                self.timer.start(50)
            except Exception as e:
                self.label_status.setText(f"Ошибка подключения: {e}")
        else:
            self.serial_port.close()
            self.serial_port = None
            self.btn_connect.setText("Подключиться к COM-порту")
            self.label_status.setText("Статус: не подключено")
            self.timer.stop()

    def update_plot(self):
        while self.serial_port and self.serial_port.in_waiting > 0:
            line = self.serial_port.readline().decode('utf-8').strip()
            #print(line)
            if line.startswith("a/g:"):
                parts = line.split()[1:]  # пропускаем "a/g:"
                print(len(parts))
                if len(parts) == 11:

                    ax, ay, az, gx, gy, gz = map(float, parts[:6])
                    acc_roll_deg, acc_pitch_deg = map(float, parts[6:8])
                    self.process_realtime_data(ax, ay, az, gx, gy, gz, acc_roll_deg, acc_pitch_deg)

        self.plot_trajectories()
        self.plot_orientation()
        self.plot_euler_angles()

    def process_realtime_data(self, ax, ay, az, gx, gy, gz, acc_roll_deg, acc_pitch_deg):
        now = time.time()
        if not hasattr(self, 'last_time_realtime'):
            self.last_time_realtime = now
            return

        dt = now - self.last_time_realtime
        self.last_time_realtime = now

        # Low-pass фильтр
        self.ax_filtered = self.ALPHA * ax + (1 - self.ALPHA) * getattr(self, 'ax_filtered', ax)
        self.ay_filtered = self.ALPHA * ay + (1 - self.ALPHA) * getattr(self, 'ay_filtered', ay)
        self.az_filtered = self.ALPHA * az + (1 - self.ALPHA) * getattr(self, 'az_filtered', az)
        self.gx_filtered = self.ALPHA * gx + (1 - self.ALPHA) * getattr(self, 'gx_filtered', gx)
        self.gy_filtered = self.ALPHA * gy + (1 - self.ALPHA) * getattr(self, 'gy_filtered', gy)
        self.gz_filtered = self.ALPHA * gz + (1 - self.ALPHA) * getattr(self, 'gz_filtered', gz)

        # Фильтр Маджвика
        self.mad_realtime.sample_period = dt
        self.mad_realtime.update([self.gx_filtered, self.gy_filtered, self.gz_filtered],
                                 [self.ax_filtered, self.ay_filtered, self.az_filtered])

        R = self.mad_realtime.get_rotation_matrix()
        acc_global = R @ [self.ax_filtered, self.ay_filtered, self.az_filtered]

        # Интегрируем RK4
        self.state_realtime = rk4_step(deriv, now, self.state_realtime, dt, acc_global)
        self.trajectory_realtime.append(self.state_realtime[:3].copy())
        self.rotations_realtime.append(R.copy())


        # Сохраняем углы Эйлера
        roll_mad, pitch_mad, yaw_mad = self.mad_realtime.get_euler_angles()
        self.roll_data.append(roll_mad)
        self.pitch_data.append(pitch_mad)
        self.yaw_data.append(yaw_mad)
        self.acc_roll_data.append(np.radians(acc_roll_deg))
        self.acc_pitch_data.append(np.radians(acc_pitch_deg))

        # Ограничиваем буфер
        for data in [self.roll_data, self.pitch_data, self.yaw_data,
                     self.acc_roll_data, self.acc_pitch_data,
                     self.trajectory_realtime, self.rotations_realtime]:
            if len(data) > self.MAX_POINTS:
                data.pop(0)

    def load_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Открыть CSV файл", "", "CSV Files (*.csv)")
        if not filename:
            return

        try:
            df = pd.read_csv(filename)

            required_columns = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("В CSV отсутствуют необходимые колонки")

            times = df['timestamp'].values
            accs = df[['ax', 'ay', 'az']].values
            gyros = df[['gx', 'gy', 'gz']].values

            # Очистка предыдущих данных CSV
            self.csv_traj = []
            self.csv_rotations = []
            self.roll_data_csv = []
            self.pitch_data_csv = []
            self.yaw_data_csv = []

            # Обработка CSV
            mad_csv = Madgwick(sample_period=1/100, beta=0.1)
            state_csv = np.zeros(6)
            traj_csv = [state_csv[:3].copy()]
            rotations_csv = []

            for i in range(1, len(times)):
                dt = times[i] - times[i - 1]
                mad_csv.sample_period = dt
                mad_csv.update(gyros[i], accs[i])

                R = mad_csv.get_rotation_matrix()
                acc_global = R @ accs[i]

                state_csv = rk4_step(deriv, times[i - 1], state_csv, dt, acc_global)
                traj_csv.append(state_csv[:3].copy())
                rotations_csv.append(R.copy())

                # Сохраняем углы из фильтра
                roll_mad, pitch_mad, yaw_mad = mad_csv.get_euler_angles()
                self.roll_data_csv.append(roll_mad)
                self.pitch_data_csv.append(pitch_mad)
                self.yaw_data_csv.append(yaw_mad)

            # Сохраняем
            self.csv_traj = np.array(traj_csv)
            self.csv_rotations = rotations_csv

            # Активируем слайдер
            self.slider.setMinimum(0)
            self.slider.setMaximum(len(self.csv_traj) - 1)
            self.slider.setValue(0)
            self.slider.setEnabled(True)
            self.slider_label.setText(f"Кадр (CSV): 0")

            drift_csv = np.linalg.norm(self.csv_traj[-1]) if len(self.csv_traj) > 0 else 0
            self.label_drift_csv.setText(f"Дрейф (из CSV): {drift_csv:.3f} м")

        except Exception as e:
            print(f"Ошибка при чтении CSV: {e}")
            self.label_status.setText(f"Ошибка CSV: {e}")

    def on_slider_change(self, value):
        self.slider_value = value
        self.slider_label.setText(f"Кадр (CSV): {value}")
        self.plot_trajectories()
        self.plot_euler_angles()

    def plot_trajectories(self):
        self.ax_traj.clear()
        self.ax_traj.set_title("Сравнение: Реальное время vs CSV")

        # Реальные данные
        if self.trajectory_realtime:
            traj_rt = np.array(self.trajectory_realtime)
            self.ax_traj.plot(traj_rt[:, 0], traj_rt[:, 1], traj_rt[:, 2], label="Реальное время", color='b')

        # CSV данные — от 0 до текущего слайдера
        if len(self.csv_traj) > 0:
            end_index = self.slider_value + 1
            traj_csv = self.csv_traj[:end_index]
            self.ax_traj.plot(traj_csv[:, 0], traj_csv[:, 1], traj_csv[:, 2], label=f"Из CSV до {end_index}", color='r', linestyle='--')

        self.ax_traj.set_xlabel("X (м)")
        self.ax_traj.set_ylabel("Y (м)")
        self.ax_traj.set_zlabel("Z (м)")
        self.ax_traj.legend()
        self.ax_traj.grid(True)
        self.canvas_traj.draw()

    def plot_orientation(self):
        if self.rotations_realtime:
            R = self.rotations_realtime[-1]
            self.ax_orient.clear()
            self.draw_orientation(R)
            self.canvas_orient.draw()

    def plot_euler_angles(self):
        self.ax_angles.clear()
        self.ax_angles.set_title("Roll / Pitch / Yaw (градусы)")
        self.ax_angles.set_xlabel("Шаг")
        self.ax_angles.set_ylabel("Угол (°)")

        # === Углы из реального времени ===
        if len(self.roll_data) > 0:
            times_rt = np.arange(len(self.roll_data))
            self.ax_angles.plot(times_rt, np.degrees(self.roll_data), 'r-', label="Roll (Realtime)")
            self.ax_angles.plot(times_rt, np.degrees(self.pitch_data), 'g-', label="Pitch (Realtime)")
            self.ax_angles.plot(times_rt, np.degrees(self.yaw_data), 'b-', label="Yaw (Realtime)")

        # === Углы из CSV ===
        if len(self.roll_data_csv) > 0:
            end_index = self.slider_value + 1
            times_csv = np.arange(end_index)
            self.ax_angles.plot(times_csv, np.degrees(self.roll_data_csv[:end_index]), 'r--', label="Roll (CSV)")
            self.ax_angles.plot(times_csv, np.degrees(self.pitch_data_csv[:end_index]), 'g--', label="Pitch (CSV)")
            self.ax_angles.plot(times_csv, np.degrees(self.yaw_data_csv[:end_index]), 'b--', label="Yaw (CSV)")

        self.ax_angles.legend()
        self.ax_angles.grid(True)
        self.canvas_angles.draw()

    def draw_orientation(self, rotation_matrix=np.eye(3)):
        origin = [0, 0, 0]
        scale = 1.0

        x_axis = rotation_matrix @ [scale, 0, 0]
        y_axis = rotation_matrix @ [0, scale, 0]
        z_axis = rotation_matrix @ [0, 0, scale]

        self.ax_orient.clear()
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
    window = IMUApp()
    window.show()
    sys.exit(app.exec_())
