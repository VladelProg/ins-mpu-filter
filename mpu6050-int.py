import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QLabel, QHBoxLayout,
    QSlider
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from madgwick_filter import Madgwick


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
# === Класс стрелок для осей ===
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        # Этот метод требуется для корректной работы с 3D графиками
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, *rest = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs3d)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, *rest = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

# === Главное окно приложения ===
class TrajectoryViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Траектория из CSV")
        self.setGeometry(100, 100, 1600, 900)

        self.init_ui()
        self.data = None
        self.trajectory = []
        self.euler_angles = []
        self.current_frame = 0
        self.animation_played = False

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)

        # Левая панель управления
        left_layout = QVBoxLayout()

        self.btn_load = QPushButton("Загрузить CSV файл")
        self.btn_load.clicked.connect(self.load_csv)
        left_layout.addWidget(self.btn_load)

        self.btn_export = QPushButton("Экспортировать график как PNG")
        self.btn_export.clicked.connect(self.export_plot)
        self.btn_export.setEnabled(False)
        left_layout.addWidget(self.btn_export)

        # Информация об углах и дрейфе
        self.label_roll = QLabel("Roll: --°")
        self.label_pitch = QLabel("Pitch: --°")
        self.label_yaw = QLabel("Yaw: --°")
        self.label_drift = QLabel("Дрейф: -- м")
        self.label_frame = QLabel("Кадр: -- / --")

        left_layout.addWidget(self.label_roll)
        left_layout.addWidget(self.label_pitch)
        left_layout.addWidget(self.label_yaw)
        left_layout.addWidget(self.label_drift)
        left_layout.addWidget(self.label_frame)

        # Слайдер для прокрутки
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.update_from_slider)
        left_layout.addWidget(self.slider)

        layout.addLayout(left_layout)

        # График траектории
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Траектория движения")
        layout.addWidget(self.canvas)

        # График ориентации
        self.fig_orientation = Figure(figsize=(4, 4))
        self.canvas_orientation = FigureCanvas(self.fig_orientation)
        self.ax_orientation = self.fig_orientation.add_subplot(111, projection='3d')
        self.ax_orientation.set_title("Ориентация объекта")
        self.draw_orientation(np.eye(3))  # начальная ориентация
        layout.addWidget(self.canvas_orientation)

        self.setCentralWidget(main_widget)

        # Однократная анимация
        self.ani = None

    def load_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Открыть CSV файл", "", "CSV Files (*.csv)")
        if not filename:
            return

        df = pd.read_csv(filename)
        times = df['timestamp'].values
        accs = df[['ax', 'ay', 'az']].values
        gyros = df[['gx', 'gy', 'gz']].values

        # Подготовка
        mad = Madgwick(sample_period=1/100, beta=0.1)
        trajectory = []
        euler_angles = []
        rotations = []
        state = np.zeros(6)

        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            mad.sample_period = dt
            mad.update(gyros[i], accs[i])

            R = mad.get_rotation_matrix()
            acc_global = R @ accs[i]

            state = rk4_step(deriv, times[i - 1], state, dt, acc_global)
            trajectory.append(state[:3].copy())
            euler_angles.append(mad.get_euler_angles())
            rotations.append(R.copy())

        self.times = times[1:]
        self.trajectory = np.array(trajectory)
        self.euler_angles = np.array(euler_angles)
        self.rotations = np.array(rotations)
        self.drift = np.linalg.norm(self.trajectory[-1])
        self.btn_export.setEnabled(True)

        # Настройка слайдера
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.trajectory) - 1)
        self.slider.setValue(0)
        self.label_frame.setText(f"Кадр: 0 / {len(self.trajectory) - 1}")

        # Очистка предыдущей анимации
        if self.ani:
            self.ani.event_source.stop()

        # Запуск однократной анимации
        self.animation_played = False
        self.ani = animation.FuncAnimation(
            self.fig, self.animate_once, frames=len(self.trajectory), repeat=False, interval=50
        )
        self.canvas.draw()

    def animate_once(self, frame):
        if self.animation_played:
            return
        self.current_frame = frame
        self.slider.blockSignals(True)
        self.slider.setValue(frame)
        self.slider.blockSignals(False)
        self.update_plot(frame)

    def update_from_slider(self):
        frame = self.slider.value()
        self.current_frame = frame
        self.update_plot(frame)

    def update_plot(self, frame):
        if not hasattr(self, 'trajectory') or len(self.trajectory) == 0:
            return

        traj = self.trajectory[:frame+1]

        # === Обновляем основную траекторию ===
        self.ax.clear()
        self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="Траектория")
        self.ax.set_title("Траектория (Маджвик + RK4)")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # === Обновляем ориентацию ===
        self.ax_orientation.clear()
        R = self.rotations[frame]
        self.draw_orientation(R)

        self.canvas.draw()
        self.canvas_orientation.draw()

        # === Обновляем метки ===
        if frame < len(self.euler_angles):
            roll, pitch, yaw = self.euler_angles[frame]
            drift = np.linalg.norm(traj[-1])
            self.label_roll.setText(f"Roll: {roll:.1f}°")
            self.label_pitch.setText(f"Pitch: {pitch:.1f}°")
            self.label_yaw.setText(f"Yaw: {yaw:.1f}°")
            self.label_drift.setText(f"Дрейф: {drift:.3f} м")
            self.label_frame.setText(f"Кадр: {frame} / {len(self.trajectory) - 1}")

    def draw_orientation(self, rotation_matrix):
        origin = [0, 0, 0]
        scale = 1.0

        x_axis = rotation_matrix @ [scale, 0, 0]
        y_axis = rotation_matrix @ [0, scale, 0]
        z_axis = rotation_matrix @ [0, 0, scale]

        self.ax_orientation.add_artist(Arrow3D([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], mutation_scale=10, lw=2, arrowstyle="-|>", color="r"))
        self.ax_orientation.add_artist(Arrow3D([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], mutation_scale=10, lw=2, arrowstyle="-|>", color="g"))
        self.ax_orientation.add_artist(Arrow3D([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], mutation_scale=10, lw=2, arrowstyle="-|>", color="b"))

        self.ax_orientation.set_xlim([-1.5, 1.5])
        self.ax_orientation.set_ylim([-1.5, 1.5])
        self.ax_orientation.set_zlim([-1.5, 1.5])
        self.ax_orientation.set_box_aspect([1,1,1])

    def export_plot(self):
        if not hasattr(self, 'trajectory'):
            return

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig_export = Figure(figsize=(10, 7))
        canvas = FigureCanvasAgg(fig_export)
        ax = fig_export.add_subplot(111, projection='3d')
        ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2])
        ax.set_title("Восстановленная траектория")
        ax.set_xlabel("X (м)")
        ax.set_ylabel("Y (м)")
        ax.set_zlabel("Z (м)")

        fig_export.savefig("trajectory_export.png", dpi=150)
        print("График сохранён как trajectory_export.png")


# === Запуск приложения ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrajectoryViewer()
    window.show()
    sys.exit(app.exec_())
