import numpy as np

class Madgwick:
    """
    Реализация AHRS алгоритма Маджвика
    https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
    """
    def __init__(self, sample_period=1/100, beta=0.1):
        self.sample_period = sample_period
        self.beta = beta  # Коэффициент коррекции гироскопа
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # qw, qx, qy, qz

    def update(self, gyroscope, accelerometer):
        """
        Обновление кватерниона по данным гироскопа и акселерометра
        """
        ax, ay, az = accelerometer
        gx, gy, gz = gyroscope

        q = self.quaternion.copy()
        qw, qx, qy, qz = q

        # Нормализация акселерометра
        norm = np.sqrt(ax * ax + ay * ay + az * az)
        if norm == 0:
            return
        ax /= norm
        ay /= norm
        az /= az

        # Вектор невязки между измеренным и оценённым ускорением
        hx = 2 * qy * qz + 2 * qw * qx - ay + 2 * qy * qy * ax
        hy = 2 * qx * qz + 2 * qw * qy - ax + 2 * qy * qz * ay
        hz = 2 * qx * qy + 2 * qw * qz - ax + 2 * qz * qz * az

        bx = np.sqrt(hx * hx + hy * hy)
        bz = np.sqrt(1 - bx * bx)

        # Градиент
        s0 = -2 * qz * ay + 2 * qy * az
        s1 = 2 * qz * ax - 2 * qw * az
        s2 = -2 * qy * ax + 2 * qw * ay
        s3 = 2 * qy * ay - 2 * qx * az

        s0 += 2 * qy * hy - 2 * qz * hz
        s1 += 2 * qz * hx - 2 * qw * hz
        s2 += -2 * qx * hy - 2 * qw * hx
        s3 += 2 * qx * hz - 2 * qy * hx

        s_norm = np.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
        if s_norm > 0:
            s0 /= s_norm
            s1 /= s_norm
            s2 /= s_norm
            s3 /= s_norm

        # Коррекция гироскопа
        gyro_correction = self.beta * np.array([s0, s1, s2, s3])
        qdot = 0.5 * np.array([
            -qx * gx - qy * gy - qz * gz,
             qw * gx + qy * gz - qz * gy,
             qw * gy - qx * gz + qz * gx,
             qw * gz + qx * gy - qy * gx
        ]) - gyro_correction

        # Интегрирование
        q += qdot * self.sample_period
        q /= np.linalg.norm(q)

        self.quaternion = q

    def get_euler_angles(self):
        """Возвращает углы Эйлера [roll, pitch, yaw]"""
        qw, qx, qy, qz = self.quaternion
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        pitch = np.arcsin(2*(qw*qy - qz*qx))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        return np.degrees([roll, pitch, yaw])

    def get_rotation_matrix(self):
        """Возвращает матрицу поворота из связанной в инерциальную"""
        qw, qx, qy, qz = self.quaternion
        return np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
            [2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
            [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]
        ])