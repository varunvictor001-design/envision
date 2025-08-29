#include <Arduino.h>

// ==========================
// Pin Definitions
// ==========================
#define BUZZER_PIN 18   // Passive buzzer GPIO
#define TRIG_PIN   13   // Ultrasonic trigger pin
#define ECHO_PIN   15   // Ultrasonic echo pin

// ==========================
// Mode: choose mock (true) or real sensor (false)
// ==========================
bool useMock = true;  // set to false to use ultrasonic sensor

// ==========================
// Setup
// ==========================
void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Initialize random seed for mock mode
  randomSeed(analogRead(0));

  // Ensure trigger pin starts LOW
  digitalWrite(TRIG_PIN, LOW);
  delay(100);
}

// ==========================
// Function: Get Real Distance
// ==========================
long getDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // timeout = 30ms
  if (duration == 0) return -1;

  long distanceCM = duration * 0.034 / 2;
  return distanceCM;
}

// ==========================
// Function: Get Mock Distance
// ==========================
long getMockDistanceCM() {
  return random(30, 71);  // 30–70 cm
}

// ==========================
// Function: Handle Distance → Buzzer
// ==========================
void handleDistance(long distance) {
  if (distance == -1) {
    Serial.println("No echo received (object too far?)");
    noTone(BUZZER_PIN);
    return;
  }

  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // Different buzzer tones for ranges
  if (distance >= 30 && distance < 40) {
    tone(BUZZER_PIN, 800, 300);   // low tone
    Serial.println("⚠️ Range: 30–40 cm (low tone)");
  } 
  else if (distance >= 40 && distance < 55) {
    tone(BUZZER_PIN, 1200, 300);  // medium tone
    Serial.println("⚠️ Range: 40–55 cm (medium tone)");
  } 
  else if (distance >= 55 && distance <= 100) {
    tone(BUZZER_PIN, 1600, 300);  // high tone
    Serial.println("⚠️ Range: 55–100 cm (high tone)");
  } 
  else {
    noTone(BUZZER_PIN);
  }
}

// ==========================
// Function: Handle Animal Sounds via Serial
// ==========================
unsigned long buzzerStart = 0;
int buzzerFreq = 0;
bool buzzerActive = false;

void handleAnimalSounds() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "tiger") buzzerFreq = 2000;
    else if (command == "lion") buzzerFreq = 2500;
    else if (command == "elephant") buzzerFreq = 500;
    else if (command == "leopard") buzzerFreq = 3000;
    else if (command == "cheetah") buzzerFreq = 4000;
    else buzzerFreq = 0;

    if (buzzerFreq > 0) {
      tone(BUZZER_PIN, buzzerFreq);
      buzzerStart = millis();
      buzzerActive = true;

      Serial.print("ANIMAL:");
      Serial.print(command);
      Serial.print(",FREQ:");
      Serial.println(buzzerFreq);

      if (command == "leopard") {
        Serial.println("🐆 Leopard Alert! Stay safe!");
      }
    }
  }

  // turn off buzzer after 1 second
  if (buzzerActive && millis() - buzzerStart >= 1000) {
    noTone(BUZZER_PIN);
    buzzerActive = false;
  }
}


// ==========================
// Main Loop
// ==========================
void loop() {
  long distance;

  // Choose mock or real
  if (useMock) {
    distance = getMockDistanceCM();
  } else {
    distance = getDistanceCM();
  }

  handleDistance(distance);

  // Check for animal sound commands
  handleAnimalSounds();

  delay(1000); // wait before next reading
}
