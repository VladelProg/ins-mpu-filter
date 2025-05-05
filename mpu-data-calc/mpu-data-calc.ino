#include <Wire.h>
#include <MPU6050_light.h>
#include <MadgwickAHRS.h>

MPU6050 mpu(Wire);
Madgwick filter;  // AHRS фильтр Маджвика

// Переменные для хранения данных
float ax = 0, ay = 0, az = 0;
float gx = 0, gy = 0, gz = 0;

// Переменные для интегрирования скорости и позиции
float vx = 0, vy = 0, vz = 0;
float x = 0, y = 0, z = 0;

unsigned long last_time = 0;
float dt = 0.01f;  // ~100 Гц

void setup() {
  Serial.begin(9600);
  Wire.begin();
  
  byte status = mpu.begin();
  Serial.print(F("MPU6050 status: "));
  Serial.println(status);
  while(status!=0){ }
  
  Serial.println(F("Калиброка датчика. Не трогайте MPU6050"));
  delay(1000);
  mpu.calcOffsets(true,true);
  Serial.println("Done!\n");
  filter.begin(100.0f);  // частота обновления фильтра
}

void loop() {
  mpu.update();         // обновляем данные
  filter.updateIMU(mpu.getGyroX(), mpu.getGyroY(), mpu.getGyroZ(),
                   mpu.getAccX(), mpu.getAccY(), mpu.getAccZ());

  // Получаем углы из фильтра
  float roll = filter.getRoll();
  float pitch = filter.getPitch();
  float yaw = filter.getYaw();

  // Интегрируем ускорение (уже в глобальной системе координат)
  unsigned long current_time = millis();
  if (last_time == 0) {
    last_time = current_time;
    return;
  }

  dt = (current_time - last_time) / 1000.0f;
  last_time = current_time;

  // Предполагаем, что акселерометр уже учёл повороты
  vx += mpu.getAccX() * dt;
  vy += mpu.getAccY() * dt;
  vz += mpu.getAccZ() * dt;

  x += vx * dt;
  y += vy * dt;
  z += vz * dt;

  // Отправляем данные на ПК
  if(millis() - timer > 10){ 
    Serial.print("data:\t");
    Serial.print(x); Serial.print("\t");
    Serial.print(y); Serial.print("\t");
    Serial.print(z); Serial.print("\t");
    Serial.print(roll * 57.2958); Serial.print("\t");   // радианы → градусы
    Serial.print(pitch * 57.2958); Serial.print("\t");
    Serial.println(yaw * 57.2958);
    timer = millis()
  }
}
