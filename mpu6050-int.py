import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QLabel, QHBoxLayout
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.animation as animation
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

# === Главное окно приложения ===
class TrajectoryViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Траектория из CSV (Улучшенная версия)")
        self.setGeometry(100, 100, 1200, 800)

        self.init_ui()
        self.data = None
        self.trajectory = []
        self.euler_angles = []

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

        left_layout.addWidget(self.label_roll)
        left_layout.addWidget(self.label_pitch)
        left_layout.addWidget(self.label_yaw)
        left_layout.addWidget(self.label_drift)

        layout.addLayout(left_layout)

        # График
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Траектория движения")

        self.setCentralWidget(main_widget)

        # Анимация
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, interval=50, cache_frame_data=False
        )

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

        self.times = times[1:]
        self.trajectory = np.array(trajectory)
        self.euler_angles = np.array(euler_angles)
        self.drift = np.linalg.norm(self.trajectory[-1])
        self.btn_export.setEnabled(True)

        # Сброс графика
        self.ax.clear()
        self.ax.set_title("Траектория (ожидание анимации...)")

    def animate(self, frame):
        if not hasattr(self, 'trajectory') or len(self.trajectory) == 0:
            return

        traj = self.trajectory[:frame+1]

        self.ax.clear()
        self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="Траектория")
        self.ax.set_title("Анимированная траектория (Маджвик + RK4)")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Обновляем метки
        if frame < len(self.euler_angles):
            roll, pitch, yaw = self.euler_angles[frame]
            self.label_roll.setText(f"Roll: {roll:.1f}°")
            self.label_pitch.setText(f"Pitch: {pitch:.1f}°")
            self.label_yaw.setText(f"Yaw: {yaw:.1f}°")
            self.label_drift.setText(f"Дрейф: {np.linalg.norm(traj[-1]):.3f} м")

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