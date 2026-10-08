// Terrain navigation vehicle controller.
// Hardware wiring and line protocol follow:
//   ~/dynamic_obstacle_ws/src/control/driving/driving.ino
//
// ROS sends: s<steering>l<left_pwm>r<right_pwm>\n
// The steering endpoint commands are configured independently in params.yaml;
// this firmware keeps the known working -7..+7 controller range.

const unsigned int MAX_INPUT = 32;

// 조향모터 드라이버
const int STEERING_IN1 = 2;
const int STEERING_IN2 = 3;
const int STEERING_POT = A0;

// 우측 구동모터 드라이버
const int RIGHT_REAR_IN1 = 8;
const int RIGHT_REAR_IN2 = 9;

// 좌측 구동모터 드라이버
const int LEFT_REAR_IN1 = 7;
const int LEFT_REAR_IN2 = 6;

const int STEERING_PWM = 128;
const int MAX_STEERING_STEP = 7;

// 가변저항 좌우측 값.  실제 차량에서 다시 읽은 뒤 이 두 값만 수정한다.
const int POTENTIOMETER_MOST_LEFT = 550;
const int POTENTIOMETER_MOST_RIGHT = 440;

const unsigned int COMMAND_INTERVAL_MS = 50;

int targetSteering = 0;
int targetLeftPwm = 0;
int targetRightPwm = 0;
unsigned long lastControlTime = 0;

void setBidirectionalMotor(int in1, int in2, int pwm) {
  pwm = constrain(pwm, -255, 255);
  if (pwm > 0) {
    analogWrite(in1, pwm);
    analogWrite(in2, 0);
  } else if (pwm < 0) {
    analogWrite(in1, 0);
    analogWrite(in2, -pwm);
  } else {
    analogWrite(in1, 0);
    analogWrite(in2, 0);
  }
}

void updateSteering() {
  const int raw = analogRead(STEERING_POT);
  const int current = map(
    raw,
    POTENTIOMETER_MOST_LEFT,
    POTENTIOMETER_MOST_RIGHT,
    -MAX_STEERING_STEP,
    MAX_STEERING_STEP
  );

  if (current < targetSteering) {
    setBidirectionalMotor(STEERING_IN1, STEERING_IN2, -STEERING_PWM);
  } else if (current > targetSteering) {
    setBidirectionalMotor(STEERING_IN1, STEERING_IN2, STEERING_PWM);
  } else {
    setBidirectionalMotor(STEERING_IN1, STEERING_IN2, 0);
  }
}

void processCommand(const char *data) {
  const char *s = strchr(data, 's');
  const char *l = strchr(data, 'l');
  const char *r = strchr(data, 'r');
  if (s == NULL || l == NULL || r == NULL || !(s < l && l < r)) {
    return;
  }

  targetSteering = constrain(atoi(s + 1), -MAX_STEERING_STEP,
                             MAX_STEERING_STEP);
  targetLeftPwm = constrain(atoi(l + 1), -255, 255);
  targetRightPwm = constrain(atoi(r + 1), -255, 255);
}

void processIncomingByte(const byte incoming) {
  static char line[MAX_INPUT];
  static unsigned int position = 0;

  if (incoming == '\n') {
    line[position] = '\0';
    processCommand(line);
    position = 0;
  } else if (incoming != '\r' && position < MAX_INPUT - 1) {
    line[position++] = incoming;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(STEERING_POT, INPUT);
  pinMode(STEERING_IN1, OUTPUT);
  pinMode(STEERING_IN2, OUTPUT);
  pinMode(RIGHT_REAR_IN1, OUTPUT);
  pinMode(RIGHT_REAR_IN2, OUTPUT);
  pinMode(LEFT_REAR_IN1, OUTPUT);
  pinMode(LEFT_REAR_IN2, OUTPUT);
}

void loop() {
  while (Serial.available() > 0) {
    processIncomingByte(Serial.read());
  }

  const unsigned long now = millis();
  if (now - lastControlTime >= COMMAND_INTERVAL_MS) {
    updateSteering();
    setBidirectionalMotor(LEFT_REAR_IN1, LEFT_REAR_IN2, targetLeftPwm);
    setBidirectionalMotor(RIGHT_REAR_IN1, RIGHT_REAR_IN2, targetRightPwm);
    lastControlTime = now;
  }
}
