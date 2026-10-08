# GNSS LiDAR Outdoor Navigation — Simulation

저장소: https://github.com/yunss01/gnss_lidar_outdoor_navigation_sim

이 저장소는 `terrain_nav_ws`의 시뮬레이션·평가 코드와 개발 이력을 관리합니다.
실차 개발은 독립 작업공간 `terrain_real_ws`와 다음 저장소에서 진행합니다:
https://github.com/yunss01/gnss_lidar_outdoor_navigation_real

기존 하드웨어 인터페이스와 관련 launch 파일은 이력 및 호환성을 위해
남겨 두었지만, 새 실차 작업공간이 이 저장소를 overlay/import하지는 않습니다.
저장소 분리 때문에 기존 패키지 이름이나 시뮬레이션 실행 명령을 바꾸지 않습니다.

GNSS 웨이포인트와 3D LiDAR를 활용한 실외 자율주행 ROS 2 워크스페이스입니다.

현재 기준 기능은 다음과 같습니다.

- GNSS 좌표 저장 및 순차 웨이포인트 주행
- F9 open route와 F10 closed lap
- 사전 점유지도 없이 rolling local costmap으로 동작하는 Nav2
- 3D LiDAR 장애물 지도, 경로 계획, 독립 긴급정지 게이트
- CARLA 외부 제어와 실차 Arduino 인터페이스의 공통 `/cmd_vel`
- YAML 기반 실차 미션 경로 입력

카메라/IMU 기반 노면 roughness 학습·추론 코드는 현재 시스템에서
제외했습니다. `terrain_mapping_node`의 경사·턱 계산은 학습 모델이 아닌
3D LiDAR 기하 처리이므로 선택 기능으로 유지합니다.

## 핵심 패키지

- `terrain_navigation_pkg`: GNSS 미션, Nav2 연결, 3D LiDAR 안전 및 시각화
- `config_pkg`: 공통 파라미터, Nav2 파라미터, YAML 미션 예제
- `launch_pkg`: 시뮬레이션·RViz·실차 실행 파일
- `vehicle_interface_pkg`: `/cmd_vel`을 Arduino 직렬 명령으로 변환
- `interfaces_pkg`: 실차 인터페이스용 사용자 정의 메시지

카메라와 2D LiDAR 패키지는 향후 사람 인식 및 센서 비교 실험을 위해
소스만 유지합니다.

## Build

```bash
git clone https://github.com/yunss01/gnss_lidar_outdoor_navigation_sim.git ~/terrain_nav_ws
cd ~/terrain_nav_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## CARLA Nav2

CARLA 서버와 `manual_control.py --external-control`을 실행한 뒤:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true
```

RViz만 별도로 실행하려면:

```bash
ros2 launch launch_pkg terrain_navigation_rviz.launch.py
```

## YAML mission route

`src/config_pkg/config/routes/mission_route_template.yaml`을 복사해 WGS84
좌표를 입력한 뒤 Nav2 런치의 `mission_route_enabled`,
`mission_route_file`, `mission_route_start` 인수로 불러옵니다.

## Safety note

실차 주행 전 실제 차폭·축거·최대 조향각, GNSS/IMU/3D LiDAR 외부
파라미터, Arduino 조향 센서 범위와 모터 방향을 반드시 다시 측정하고
저속 벤치 테스트해야 합니다.
