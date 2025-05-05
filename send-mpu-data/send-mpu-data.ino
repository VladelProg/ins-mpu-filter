#include "Wire.h"
#include <MPU6050_light.h>

MPU6050 mpu(Wire);

long timer = 0;

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
}

void loop() {
  mpu.update();

  if(millis() - timer > 10){ // отправка данных каждые 10 мс
    Serial.print("a/g:\t"); // Маркер начала сообщения
    // Значение акселерометра
    Serial.print(mpu.getAccX()); Serial.print("\t");
    Serial.print(mpu.getAccY()); Serial.print("\t");
    Serial.print(mpu.getAccZ()); Serial.print("\t");

    // Значение гироскопа
    Serial.print(mpu.getGyroX()); Serial.print("\t");
    Serial.print(mpu.getGyroY()); Serial.print("\t");
    Serial.print(mpu.getGyroZ()); Serial.print("\t");
    
    // Roll/Pitch/ Yaw
    Serial.print(mpu.getAccAngleX()); Serial.print("\t");
    Serial.print(mpu.getAccAngleY()); Serial.print("\t");
    
    Serial.print(mpu.getAngleX()); Serial.print("\t");
    Serial.print(mpu.getAngleY()); Serial.print("\t");
    Serial.println(mpu.getAngleZ()); Serial.print("\t");
    timer = millis();
  }
}
