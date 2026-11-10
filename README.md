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

## 직선 회피·복귀 / 조향·위치 오차 실험 (2026-10-13)

이 실험은 기존 F9/F10과 별개다. `scripts/experiment.sh`를 **실행**하면
현재 터미널 설정을 바꾸지 않고 ROS 도메인 **113**에서 동작한다.
일반 시뮬레이션 112, 실차 42와 통신하지 않는다. 단, CARLA 서버는 같은
2000 포트를 사용하므로 **기존 Pygame/manual_control과 Nav2는 종료**하고 시작한다.
실차 ws, 기본 센서·Nav2 YAML, 외부 `manual_control.py`는 수정하지 않는다.
실험 Pygame의 저장 경로도 `~/terrain_nav_data/experiments/ui`로 분리해,
기존 F9/F10 웨이포인트 파일을 읽거나 덮어쓰지 않는다.

실험 실행기는 기존 CARLA 브리지와 제어 메서드를 해당 프로세스 안에서만
감싼다. 안전 게이트는 최종 `/cmd_vel`의 유일한 발행자로 유지된다.
`run` 명령이 준비 검사를 통과하기 전에는 차량이 움직이지 않고, 종료·중단 후에는
실험 클라이언트가 잔여 Nav2 명령을 거부하고 원래 제동 경로로 정지한다.
**실험에서는 F9/F10을 누르지 말고 아래 `run` 명령으로 출발한다.**

### 최초 1회 빌드

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh colcon build \
  --packages-select terrain_navigation_pkg launch_pkg --symlink-install
```

아래 명령은 모두 `~/terrain_nav_ws`에서 실행한다. `experiment.sh`가 ROS와
이 ws를 읽으므로 터미널마다 별도로 `source`할 필요는 없다.

### 첫 시험: S0 / baseline

1. 터미널 A에서 CARLA 서버를 실행하고 로딩을 기다린다. 이미 서버가 켜져 있으면
   다시 실행하지 않는다.

   ```bash
   bash ~/carla/0.10.0/CarlaUnreal.sh
   ```

   Town10HD_Opt가 아닌 맵에서 적절한 직선을 찾지 못한다면, **클라이언트를 켜기 전**
   다른 터미널에서 다음을 실행한다. 이 명령은 CARLA 월드를 교체하며 기존 actor를
   제거하므로 진행 중인 장면에는 실행하지 않는다. 저장된 코드/기록은 삭제하지 않는다.

   ```bash
   bash scripts/experiment.sh python3 \
     ~/carla/0.10.0/PythonAPI/util/config.py --map Town10HD_Opt
   ```

2. 터미널 B에서 정지 상태의 실험 클라이언트를 켠다.

   ```bash
   bash scripts/experiment.sh python3 scripts/experiment_manual.py \
     --profile baseline --auto-straight-start --gnss-quality rtk_fixed
   ```

   이 옵션은 40m 전방이 Driving lane이고 교차로가 아닌 후보를 고른다.
   **장애물 없는 도로나 좌우 공간을 보장하지는 않는다.** Pygame에서 평탄한 직선과
   우회 공간을 확인한다. 부적합하면 클라이언트만 종료하고 다음 후보를 고른다:

   ```bash
   bash scripts/experiment.sh python3 scripts/experiment_manual.py \
     --profile baseline --auto-straight-start --start-candidate 2 \
     --gnss-quality rtk_fixed
   ```

   후보를 읽기만 하려면 `python3 -m terrain_navigation_pkg.simulation_experiment
   find-start`를 `experiment.sh`로 실행한다. NPC를 자동으로 생성하거나 기존 actor를
   일괄 삭제하지 않는다. 첫 시험에서는 주변 움직이는 교통이 없는 장면을 사용한다.

3. 터미널 D에서 고정 시작점/40m 목표를 저장하고 S0를 준비한다.

   ```bash
   bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
     capture --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json

   bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
     prepare --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
     --scenario S0
   ```

   `capture`는 기존 anchor를 덮어쓰지 않는다. 다른 장소는 `straight_v2`처럼 새 이름을
   사용한다. 출발점은 CARLA 차량 원점이고, 기본 장애물 중심은 그곳에서 전방20m다.
   A/B의 GNSS 좌표는 부착된 안테나 위치에서 생성해 기존 GNSS 원점 계약과 맞춘다.
   실제 이동·복귀·목표 오차는 차량 원점 기준으로 평가하여 안테나 오프셋을 혼동하지 않는다.

4. 터미널 C에서 실험 전용 Nav2를 켠다.

   ```bash
   bash scripts/experiment.sh ros2 launch launch_pkg simulation_experiment.launch.py
   ```

   RViz용 표시 노드를 제외하려면 `viz:=false`를 덧붙인다. 센서, Nav2와 모든 costmap은
   같은 CARLA `/clock`을 사용한다. 기본 속도 설정은 기존 약4km/h를 유지한다.
   클라이언트의 20FPS 상한은 빠른 PC에서 시뮬레이션만 급히 진행하는 것을 방지한다.

   점군·경로·costmap을 보려면 별도 터미널 E에서 RViz를 실행한다:

   ```bash
   bash scripts/experiment.sh rviz2 \
     -d src/launch_pkg/rviz/terrain_navigation.rviz \
     --ros-args -p use_sim_time:=true
   ```

5. 터미널 D에서 출발·평가를 시작한다.

   ```bash
   bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
     run --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
     --scenario S0 --profile baseline
   ```

   준비 검사에는 활성 Nav2 lifecycle, 최신 LiDAR, GNSS 원점, 실험 클라이언트,
   단일 제어용 odometry/TF/최종명령 발행자와 odometry-TF 일치가 포함된다.
   미충족이면 출발하지 않고 `preflight_failed`로 기록한다.

### 다음 시험: S1 및 오차 비교

매 시험마다 **터미널 C의 Nav2와 터미널 B의 클라이언트를 종료하고 재시작**한다.
출발점·물리 상태와 costmap/계획기 상태를 섞지 않기 위해서다. CARLA 서버는 유지한다.
이 방식은 조건을 고정하지만 월드 재로드까지 포함한 bit-identical 재현성은 주장하지 않는다.
순서는 **차량 정지 확인 → C 종료 → B를 켜둔 채 D에서 cleanup → B 종료/재시작**이다.
동기식 CARLA의 시계를 진행하는 B를 먼저 끄면 cleanup이 tick을 기다리다 실패할 수 있다.

1. 터미널 D에서 이전 시험이 만든 장애물만 정리한다.

   ```bash
   bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
     cleanup --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json
   ```

   actor ID, blueprint, 실험 고유 role을 모두 확인한다. 다른 actor는 제거하지 않으며
   준비·시험 기록은 보존한다. 제거한 장애물은 다음 `prepare`로 다시 만들 수 있다.

2. 터미널 B에서 **동일 anchor**로 다시 켠다. 기록된 차량 blueprint도 자동으로 고정한다.

   ```bash
   bash scripts/experiment.sh python3 scripts/experiment_manual.py \
     --profile baseline --gnss-quality rtk_fixed \
     --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json
   ```

3. 터미널 D에서 S1을 준비한다. 중앙에 주차된 차량을 고정 장애물로 생성한다.
   이번 단계는 작은 물체 인지가 아니라 회피·복귀 시험이다.

   ```bash
   bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
     prepare --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
     --scenario S1
   ```

4. 터미널 C에서 같은 실험 Nav2 런치를 켜고, D에서 실행한다.

   ```bash
   bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
     run --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
     --scenario S1 --profile baseline
   ```

그다음 같은 절차에서 **B와 D의 `--profile`을 함께** 바꾼다.
위 명령의 `rtk_fixed`는 저잡음 GNSS 가정이며 실제 수신기의 성능 검증을 대신하지 않는다.
초기 회피 시험의 GNSS 원점 흔들림을 줄이기 위해 사용하고, 비교 중에는 계속 동일하게 유지한다.
사용한 센서 profile과 초기 GNSS 원점은 각 시도의 `metadata.json`에 저장된다.

| profile | 주입 조건 |
|---|---|
| baseline | 연속 조향, 위치·방향 편향 없음 |
| steering_steps | 정규화 조향 -7~7의 15단계 |
| steering_delay | 조향 응답만 0.2초 지연 |
| steering_both | 단계 제한 + 조향 지연 |
| position_left / position_right | 경로 기준 좌우 위치 ±0.3m 편향 |
| yaw_left / yaw_right | 방향 ±3도 편향 |

주입은 출발3초 뒤 켜진다. 위치·방향 편향은 이후1초 동안 증가하며, GNSS 원점과
목표 경로를 같이 움직여 편향이 상쇄되는 것을 방지한다. 위치 오차는 제어용
`/vehicle/odometry`와 유일한 `odom -> base_link` TF에 동일하게 적용한다.
별도의 `/experiment/truth_odometry`에는 TF 권한이 없으며 평가에만 사용한다.
조향 지연에 제동·정지명령·watchdog를 포함하지 않는다.

**현재 위치는 여전히 CARLA 참위치를 바탕으로 만든 합성 위치다.** 이것은 GNSS/IMU
융합이나 실차 조향각 보정의 성공을 뜻하지 않는다. 조향 단계도 ADC/실제 바퀴각이
아니라 CARLA 정규화 조향의 시험용 양자화다. `pose_fault_observed`와
`steering_effect_observed`로 실제 기록에 오차/조향 변화가 나타났는지도 확인한다.
S0처럼 조향 요청이 거의 없는 시험은 조향 오차 모델을 충분히 자극하지 못할 수 있다.

S0와 S1을 먼저 한 번씩 완주해 도구를 확인하고, 그 후 같은 조건을 3~5회 반복한다.
오차를 추가하기 전 baseline이 실패하면 그 실패부터 분석한다. 개선 버전과 비교할 때는
같은 anchor·차량·센서·속도·장애물 조건을 사용하고 실패 시도도 남긴다.

### 2026-10-20 재개: S1 도착 반경 3m / 1m 비교

이번에는 F9/F10을 누르지 않는다. **D의 `run`이 준비 검사 후 출발시킨다.**
기존 S1은 참위치 목표 약2.7m 전에서 경로가 완료되고 정지 후 방향 오차12.5도가
남았다. GNSS 도착 반경이 복귀를 조기에 끊었는지 확인하는 시험이지,
반경1m만으로 복귀가 해결됐다는 주장이 아니다. 조향/위치 편향은 아직 추가하지 않는다.

실험 전용 임시 YAML에서 GNSS `arrival_radius_m`만 바꾼다. 일반 F9/F10의 공통
YAML, Nav2 XY/yaw 허용 오차, 속도, 안전 게이트와 아래 복귀 합격 기준은 유지한다.
허용 반경은1~3m이며 기본값은3m다. **C와 D에 같은 반경을 지정해야 한다.**
D는 실제 `/gnss_goal_manager_node/get_parameters`를 읽어 다르면 출발을 거절한다.
이 값은 실행 중 `ros2 param set`으로 바꾸지 말고, C를 재시작해 적용한다.

먼저 한 번 빌드한다. 어느 터미널이든 가능하다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh colcon build \
  --packages-select terrain_navigation_pkg launch_pkg --symlink-install
```

터미널 A — CARLA 서버. 이미 실행 중이면 중복 실행하지 않는다.
맵은 저장된 anchor와 같은 `Town10HD_Opt`여야 한다. 맵 불일치 오류가 나면 중단하고
확인한다. 무턱대고 월드를 재로드하여 다른 actor를 지우지 않는다.

```bash
bash ~/carla/0.10.0/CarlaUnreal.sh
```

터미널 B — 매 시험 동일한 명령. 다른 manual_control 클라이언트는 종료한 뒤 실행한다.
`--anchor`가 기존 S1의 차량과 x/y/yaw를 복원한다. 별도 `--filter`,
`--spawn-point`, `--auto-straight-start`는 붙이지 않는다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 scripts/experiment_manual.py \
  --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --profile baseline \
  --gnss-quality rtk_fixed \
  --res 1280x720 --resizable \
  --semantic-lidar-labels
```

기록된 차량은 `vehicle.lincoln.mkz`, CARLA 좌표는
`x=-106.11748504638672, y=-4.4253716468811035, yaw=-91.46309661865234°`다.
스폰 높이는 기록된 지면 정착 높이보다0.25m 위로 주어 물리적으로 내려앉힌다.
이번 anchor는 과거 일반 F9용 `63.508,68.225,...` 위치와 다르다.
차량이 내려앉은 뒤 손으로 이동시키거나 키보드로 주행하지 않는다.

터미널 D — B의 차량 화면이 보이는 상태에서 준비한다. 이전 실험 장애물만 정리한 뒤
동일한20m 중앙 장애물을 다시 만든다. 기록 파일과 관계없는 actor는 보존한다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  cleanup --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json

bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  prepare --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1
```

`Prepared S1: 1 frozen test vehicles`는 준비 완료다. 차량 출발 명령은 아니다.
Pygame/점군에서 장애물 배치를 확인한다. 기존 C가 있다면 먼저 Ctrl+C로 종료한다.

터미널 C — 첫 시험의 기준 반경3m.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh ros2 launch launch_pkg simulation_experiment.launch.py \
  arrival_radius_m:=3.0
```

터미널 D — C가 켜진 뒤 출발·기록한다. 준비 검사는 최대45초 기다린다.

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  run --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1 --profile baseline --arrival-radius-m 3.0
```

`Preflight OK: GNSS arrival radius 3 m verified`를 확인한다.
불일치/센서/TF/활성화 오류는 우회하지 않는다. D 오류로 F9를 대신 누르면 안 된다.

둘째 시험 — A는 유지한다. D 종료·차량 정지 확인 후 C를 Ctrl+C로 끄고,
**B가 켜진 상태에서** D의 위 `cleanup`을 실행한다. B를 종료하고 위 B 명령으로
재시작한 뒤 D의 위 `prepare ... --scenario S1`을 다시 실행한다.
그다음 C/D만 다음 두 명령으로 바꾼다.

```bash
# 터미널 C
bash scripts/experiment.sh ros2 launch launch_pkg simulation_experiment.launch.py \
  arrival_radius_m:=1.0
```

```bash
# 터미널 D
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  run --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1 --profile baseline --arrival-radius-m 1.0
```

결과는 매번 새 `~/terrain_nav_data/experiments/runs/run_.../`에 남는다.
두 시험의 D 마지막 JSON과 `Trial output:` 경로를 함께 비교한다.
우선 3m/1m 한 쌍으로 원인을 분리하고, 개선이 보이면 각각3회 이상 반복한다.
비교 순서도 번갈아 시행한다. 실패 시도를 지우거나 성공 결과로 대체하지 않는다.

```bash
# 터미널 D: 모든 시도와 조건별 묶음을 새 CSV로 내보낸다.
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment report
```

새 기록에는 요청/실제 반경과 `arrival_radius_verified`를 저장하며,
report는 anchor·클라이언트·코드 해시와 요청/실제 반경별로 묶는다.
기존 기록에 반경 정보가 없으면 미확인으로 남기고3m라고 추측하지 않는다.
코드가 달라진 과거3m 결과만 대조군으로 쓰지 말고, 이번 코드로3m부터 새로 실행한다.

판정에서 `goal_arrived`는 기존 **차량 참위치 기준 목표3m 이내**라는 뜻이다.
이 값을1m 진입 증거로 읽으면 안 된다. 새 진단값
`navigation_goal_reached_observed`는 최종 목표 근처(참위치3m 이내)에서의
GNSS 도착 신호 수신 여부,
`minimum_navigation_goal_distance_m`는 같은 구간에서 최신 GNSS 목표거리의 최솟값,
`route_completed_observed`는 경로 완료 수신 여부다. GNSS는 안테나, 참위치는 차량
원점 기준이므로 두 거리에는 부착 오프셋 차이가 있다.
시작점 A에서도 첫 waypoint 도착 신호가 발생하므로 그 신호/거리0m는 최종 B 도착
진단에서 제외한다.
이 진단값들은 합격 기준을 바꾸지 않는다. 도착 신호 없이 가까이서 멈춘 경우는
반경1m가 의도대로 경로 종료를 제어한 증거로 취급하지 않는다.

`metrics_valid=true`, 무충돌/통로 유지와 함께 `recovered=true`인지 본다.
도착해도 `recovered=false`면 복귀 실패다. 1m에서도 방향 복귀가 안 되면 반경을
더 줄이거나 방향 허용치를 키우지 말고 장애물 뒤 복귀 경로·진입 방향을 분석한다.
아래 개발 검사는 CARLA의 새 실제 주행 성공을 대신하지 않는다.

### 2026-10-20 다음 단계: 1m baseline / rejoin 비교

3m/1m 각3회 결과는0/3 대2/3 통과였다. 1m 성공2회도 경로 완료 후 잔여 이동 중
방향5도 이내로 들어왔으므로 안정적인 주행 복귀가 해결된 것은 아니다.
이제 **반경은1m로 고정**하고 `navigation_variant`만 비교한다.
기존 `--profile baseline`은 오차 주입 없음이라는 별개 조건이며 계속 유지한다.

- `baseline`: 기존 FAR 목표 선택. 복귀 지점을 추가하지 않는다.
- `rejoin`: 현재 mission의 진행 방향을 이용해 **최종 목표8m 전 기준선상의 가상
  복귀 지점**을 먼저 계획한다. 이40m 시험에서는 약32m 지점이다.
  그 지점을 Nav2 목표로 보낼 때 방향을 원래 진행 방향으로 지정한다.
  지점1m 전 평면에 도달하면 최종 목표로 넘어가고, 이미 지난 지점을 뒤로 요청하지 않는다.
  이 단계 전환 자체가 실제 복귀 성공 판정은 아니다.

제어 입력은 기존 navigation odometry/GNSS route tangent이며 평가용 참위치나
장애물 fixture 좌표를 계획기에 넣지 않는다. Smac/Dubins, 경로 효율 검사,
경로 안전 게이트와 원시 LiDAR 비상정지는 유지한다. 막힌 복귀 지점을 옆으로
자동 옮기거나 장애물 셀을 지워 통과시키지 않는다. 짧은 구간의 방향 제약이
불가능하면 기존 효율/안전 검사에서 거절될 수 있다. 이때 검사를 끄지 말고 결과를 분석한다.
일반 F9/F10에서는 새 복귀 옵션이 기본0m(비활성)이며 공통 YAML은 변경하지 않았다.

먼저 빌드한다. 실행 중인 C는 종료해야 새 코드/파라미터가 적용된다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh colcon build \
  --packages-select terrain_navigation_pkg launch_pkg --symlink-install
```

터미널 A의 CARLA 서버는 그대로 유지한다. F9/F10은 누르지 않는다.
매 시험의 초기화는 **차량 정지 확인 → C 종료 → B 유지한 채 D cleanup → B 종료 →
B 동일 anchor로 재시작 → D prepare → C → D run**이다.
경고가 난 경우에도 클라이언트를 계속 재사용하지 않는다.

터미널 D: 기존 B가 켜진 상태에서 이전 실험 장애물만 정리한다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  cleanup --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json
```

이후 B를 종료하고 터미널 B에서 다음을 실행한다. **이 명령은 두 조건에서 동일**하다.
같은 Lincoln과 기존 S1의 x/y/yaw를 anchor로 복원한다.
별도 `--spawn-point`, `--filter`, `--auto-straight-start`를 덧붙이지 않는다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 scripts/experiment_manual.py \
  --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --profile baseline --gnss-quality rtk_fixed \
  --res 1280x720 --resizable --semantic-lidar-labels
```

터미널 D: 새 B의 차량 화면이 보인 뒤 준비한다. 준비 후 B를 재시작하면 다시 준비해야 한다.

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  prepare --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1
```

첫 시험은 새 코드의 대조군 `baseline`이다. 종료 확인/계측 코드도 바뀌었으므로
과거1m 결과만 대조군으로 대체하지 않는다.

```bash
# 터미널 C
cd ~/terrain_nav_ws
bash scripts/experiment.sh ros2 launch launch_pkg simulation_experiment.launch.py \
  arrival_radius_m:=1.0 navigation_variant:=baseline
```

```bash
# 터미널 D
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  run --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1 --profile baseline \
  --arrival-radius-m 1.0 --navigation-variant baseline
```

둘째 시험은 위 초기화/cleanup/B 재시작/prepare를 그대로 반복한 뒤 다음으로 바꾼다.

```bash
# 터미널 C
bash scripts/experiment.sh ros2 launch launch_pkg simulation_experiment.launch.py \
  arrival_radius_m:=1.0 navigation_variant:=rejoin
```

```bash
# 터미널 D
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  run --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1 --profile baseline \
  --arrival-radius-m 1.0 --navigation-variant rejoin
```

D는 GNSS 반경뿐 아니라 실제 FAR 노드의 복귀 거리/전환 거리 파라미터를 조회한다.
C/D 조건이 다르거나 예전 C가 실행 중이면 출발을 거절한다. 실행 중 `ros2 param set`으로
바꾸지 말고 C를 재시작한다. 설정은 metadata/summary/조건별 report에 구분해 남긴다.

새 결과에서 다음을 함께 확인한다.

| 필드 | 의미 |
|---|---|
| `navigation_variant_verified` | C의 실제 복귀 설정이 D와 일치함 |
| `terminal_approach_observed` | 복귀 지점 계획 단계가 관측됨. baseline은false가 정상 |
| `motion_success` | 기존 주행 판정. 횡0.5m/방향5도/연속2초 기준은 그대로 |
| `moving_recovery_before_route_completion` | 장애물 뒤에서 속도0.2m/s 이상으로 정렬을2초 유지하고 경로 완료 전에 확인됨 |
| `shutdown_confirmed` | 같은 run/session의 새 종료 응답과 이후 새 정지 측정을 확인함 |
| `success` | 기존 주행 합격과 새 종료 확인을 모두 만족함 |

`rejoin`에서 기존 `success=true`만 나오고 주행 중 사전 복귀 지표가false라면,
새 설계 목표 달성으로 보지 않는다. 반대로 `terminal_approach_observed=true`만으로
복귀 성공이라고 하지 않는다. 최종 목표 위치·충돌·간격·경로 이탈·시간도 함께 본다.
우선 한 쌍의 마지막 JSON과 `Trial output:` 경로를 확인하고, 개선이 보이면 동일 조건으로
3회씩 교대 반복한다. 기존 실패와 종료 경고가 있는 원본 결과는 지우거나 수정하지 않는다.

종료 확인은 기존30콜백 방식 대신 **최대5초 벽시계 시간** 동안 stop을 재전송하며,
같은 run/session의 새 `armed=false`와 그 이후 새 odometry에서0.05m/s 이하를
시뮬레이션0.1초 이상 확인한다. 오래된 정지 캐시만으로 확인하지 않는다.
실패하면 경고와 `shutdown_confirmed=false`를 남기고 전체 성공으로 승인하지 않는다.
주행 자체가 통과했더라도 `verdict=shutdown_unconfirmed`, `success=false`가 된다.
그 경우 B를 종료·재시작한다. 과거 파일에는 새 종료 판정을 소급 적용하지 않는다.

개발 검사는 단위/모의 geometry와 격리 ROS 통신 검사다.
**실제 CARLA에서 새 rejoin 조건이 통과했다는 뜻은 아니며 위 주행으로 확인해야 한다.**

### 2026-10-20 후속: rejoin 고정, legacy/latest 명령 수신 비교

이 절의 legacy/latest 각3회 비교는 완료했다. **현재 다음 시험은 아래의
"latest S1 경로/추종 계측" 절을 사용한다.** 직전 rejoin은 주행 중 복귀했지만
정지할 때 방향이 다시5도를 넘어 최종8.06도가 됐다. 당시 route 완료 뒤 전제동
관측까지0.85초/정지까지1.00초가 걸렸고 약0.85m 이동했다. 경로 완료와 실제 제동은
동일 시점이 아니다. 이 값에는 상위 속도 감속과 기록 시차도 포함되므로 전부를
네트워크 지연이라고 부르지 않는다.

원본 클라이언트는 프레임당 ROS 콜백 하나를 처리하고 속도 명령을 depth10에 보관한다.
이번에는 **경로/반경은 그대로 두고 수신만 비교**한다. 일반 manual_control.py,
F9/F10, 실차, Nav2, 안전 게이트, 공통 YAML은 이번 후속 변경에서 수정하지 않는다.

- `legacy`: 기존 수신 동작. 새 계측만 추가한 대조군. B/D의 기본값도legacy다.
- `latest`: 기존 command subscription을 제거하고, KEEP_LAST depth1인 전용 노드/
  executor에서 제어 직전에 최신 명령 한 개를 처리한다. 스레드나 두 번째 명령
  발행기를 추가하지 않는다. superseded setpoint만 버리며 스로틀/조향/제동 계산은
  원본 제어기를 사용한다. 명령 없음/수신 후0.5초 단절 시 기존 전제동도 유지한다.
- command mode는 B와D에서만 바꾼다. C는 두 시험 모두 `navigation_variant:=rejoin`이다.
  `--profile baseline`도 계속 유지한다. 이것은 수신 모드와 다른 오차 주입 조건이다.
- 새 B는 모드/계측 버전을 발행한다. D가 실제 값을 확인하고 다르면 출발 전에 거절한다.
  이전 버전 B를 그대로 사용하면 계측 불일치로 차단되므로 B도 반드시 재시작한다.

수정 후 빌드(이번 구현 작업에서 이미 빌드했더라도 재실행 가능):

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh colcon build \
  --packages-select terrain_navigation_pkg launch_pkg --symlink-install
```

터미널 A: 기존 CARLA 서버를 유지한다. 꺼져 있을 때만 다음을 실행한다.

```bash
bash ~/carla/0.10.0/CarlaUnreal.sh
```

매 시험 초기화 순서는 **정지 확인 → C Ctrl+C → 기존 B를 켜둔 채 D cleanup →
B 종료 → 새 B → D prepare → C → D run**이다. 동기식 월드에서 유일한 B를 먼저
끄면 cleanup이 기다릴 수 있다. F9/F10은 사용하지 않는다.

터미널 D, 기존 B가 아직 켜진 상태에서:

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  cleanup --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json
```

그다음 B를 종료한다. 첫 S1 대조군을 위해 터미널 B에서:

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 scripts/experiment_manual.py \
  --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --profile baseline --command-mode legacy --gnss-quality rtk_fixed \
  --res 1280x720 --resizable --semantic-lidar-labels
```

기존 anchor의 같은 Lincoln/위치/방향을 복원한다. 다른 spawn/filter 옵션을 붙이지 않는다.
터미널 D에서 새 차량을 확인한 뒤:

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  prepare --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1
```

터미널 C, **양쪽 수신 조건에서 동일**:

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh ros2 launch launch_pkg simulation_experiment.launch.py \
  arrival_radius_m:=1.0 navigation_variant:=rejoin
```

터미널 D, 첫 S1 `legacy` 실행:

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  run --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1 --profile baseline --arrival-radius-m 1.0 \
  --navigation-variant rejoin --command-mode legacy
```

끝나면 위 초기화를 처음부터 반복한다. 둘째 S1 `latest`의 B:

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 scripts/experiment_manual.py \
  --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --profile baseline --command-mode latest --gnss-quality rtk_fixed \
  --res 1280x720 --resizable --semantic-lidar-labels
```

D의 `prepare --scenario S1`과 C의 동일 명령을 다시 실행한 뒤, D:

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  run --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1 --profile baseline --arrival-radius-m 1.0 \
  --navigation-variant rejoin --command-mode latest
```

첫 latest 주행 전에 장애물 없는 S0 점검1회를 권장한다. 새 B는latest, C는 위와 같고,
prepare/run 양쪽 `--scenario S1`만 `--scenario S0`로 바꾼다. 정상 주행/정지,
`metrics_valid`, `success`, `command_mode_verified`, `shutdown_confirmed`가true인지
확인한다. S0의 `aligned_stop_success=null`과 `recovered=false`는 실패 근거가 아니다.
경고/계측 불일치가 있으면 S1로 진행하지 말고 해당 JSON과 출력을 확인한다.
S0 뒤에도 전체 초기화와 S1 prepare를 다시 한다.

S1은 먼저legacy/latest 한 쌍을 비교하고, 계측·안전이 정상일 때
`legacy → latest → legacy → latest → legacy → latest`로 각각3회까지 교대한다.
기존 코드/수신 버전 결과를 새 대조군으로 대신하지 않는다. 모든 실패/정지 경고도 남긴다.

이번 목표의 별도 강화 진단은 다음과 같다. **기존 success/pass/recovered는 바꾸지 않는다.**

| 필드 | 확인할 내용 |
|---|---|
| `command_mode` / `command_mode_verified` | 실제 B 수신 방식과 D 요청 일치 |
| `moving_recovery_before_route_completion` | 기존 횡0.5m/방향5도/2초/속도0.2m/s 조건의 조기 복귀 |
| `recovery_retained_until_stop` | 첫 주행 중 복귀 확인 이후 기록 끝까지 정렬을 벗어나지 않고 정지함 |
| `final_alignment_within_bounds` | 최종 횡0.5m/방향5도 이내이며 속도0.05m/s 이하 |
| `aligned_stop_success` | S1/S2에서 기존 전체success와 정지까지 정렬 유지를 함께 만족 |
| `completion_to_accepted_stop_s` | route 완료 후 실제 소비한0속도 명령이 처음 기록될 때까지. 과거 파일은null |
| `completion_to_full_brake_s` | route 완료 후 전제동이 처음 기록될 때까지 |
| `completion_to_standstill_s` / `post_completion_distance_m` | route 완료 후 정지까지 시간과 이동 거리 |
| `maximum_heading_error_after_route_completion_deg` | 최종값만 좋아져도 중간 재이탈을 숨기지 않음 |

`aligned_stop_success`의 기존 위치 합격 계약은 참위치 목표3m 이내 그대로다.
GNSS 제어의1m와 참위치(body) 목표 거리 기준을 혼동하지 않는다. 위치 오차/최소 간격,
충돌/통로 이탈, 소요시간/정지시간도 함께 비교한다. latest가 제동 지연을 줄여도
주행 중 복귀나 간격을 악화시키면 개선 완료로 선언하지 않는다.

CSV에는 차량에서 실제 소비한 `accepted_command_speed_mps`,
`accepted_command_yaw_rate_rps`, `accepted_command_wall_age_s`와
`control_sequence`, `control_sim_time_s`, `command_rx_count`, `throttle`을 추가했다.
소비 명령은 마지막 제어 호출의 관측값이다. None/빈값이면 명령 없음/미계측이지
0속도 명령을 받았다는 뜻이 아니다. 수신 나이는 **콜백 수신 이후** 시간이며,
Twist에는 발행 timestamp가 없으므로 절대적인 발행→제동 지연이라고 해석하지 않는다.
기록은 샘플 간격/상태 시차가 있어 정확한 지연은 CSV와 함께 판단한다.

반증 기준: latest에서도 실제 소비한0명령 뒤 제동이 늦거나 최종 정렬이 계속 실패하면,
이번 수신 개선만으로 해결되지 않은 것이다. 다음 변경은 종단 경로/정지 제어를
따로 분리해 설계한다. 도착 반경 축소·방향 기준 완화로 통과시키지 않는다.
반복 결과의 마지막 JSON과 `Trial output:` 경로를 전달하면 원시 구간까지 비교할 수 있다.

### 2026-10-20 다음 시험: latest S1 경로/추종 계측

이 절의 첫 계측 주행은 `run_20261020_154142_150a2824`로 완료했다.
**지금 최소 보완 검증에는 아래 "계측 v2" 절의 오프라인 재분석을 사용한다.
같은 S1을 다시 주행할 필요는 없다.** 이 절의 B/C/D 명령은 향후 새 주행용으로 유지한다.

legacy/latest 각3회는 모두 주행·종료 판정을 통과했지만, 정지까지 정렬 유지는
legacy2/3회, latest3/3회였다. latest의 route 완료 후 이동은 평균0.33m로 줄었지만
최소 물리 OBB 간격은 각 대응 시험에서 더 작았다(legacy 평균0.758m,
latest0.623m). 아직 경로 추종 문제라고 단정할 수 없다. 기존 `cross_track`은
시작 직선 기준 이탈이므로 실제 계획 경로 추종 오차를 뜻하지 않는다.

**이번에는 주행 파라미터를 바꾸지 않고 latest S1을 계측 포함1회 실행한다.**
`rejoin`, 도착 반경1m, 기존 anchor/Lincoln/S1 배치를 그대로 사용한다.
cost_penalty/lookahead/속도/footprint/안전 게이트는 변경하지 않는다.
기존 F9/F10, 일반 manual_control.py와 실차 실행도 바꾸지 않는다.

추가 계측은 opt-in `--trace-paths`다. `/plan` 후보와 RPP 계산 경로를 구분하고,
원본 pose/frame/header·수신시각·FollowPath goal 상태·정확한 시각의 TF·LiDAR 여유
상태를 기록한다. Humble의 `/received_global_plan`은 이름과 달리 차량 좌표계로
변환·잘린 계산 경로이며, 원본 전체 경로나 실제 조향 적용의 증거가 아니다.
([Nav2 Humble RPP 공식 소스](https://github.com/ros-navigation/navigation2/blob/humble/nav2_regulated_pure_pursuit_controller/src/regulated_pure_pursuit_controller.cpp))
goal ID는 상태 수신 시점의 연결 근거이며, 후보 ID는 경로 수락 ID로 취급하지 않는다.
목표 전환 구간에서 연결 근거가 부족하면 정상 추종으로 추정하지 않는다.

빌드:

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh colcon build \
  --packages-select terrain_navigation_pkg launch_pkg --symlink-install
```

터미널 A는 기존 CARLA 서버를 유지한다. 꺼졌을 때만:

```bash
bash ~/carla/0.10.0/CarlaUnreal.sh
```

초기화는 **정지 확인 → C Ctrl+C → 기존 B를 켜둔 채 D cleanup → B 종료** 순서다.
유일한 동기식 B를 먼저 종료하면 cleanup이 기다릴 수 있다. D:

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  cleanup --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json
```

기존 B를 종료한 뒤 터미널 B에서 다음을 실행한다. anchor에서 이전과 같은
Lincoln/스폰 위치/방향을 복원하므로 별도 spawn/filter 옵션을 추가하지 않는다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 scripts/experiment_manual.py \
  --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --profile baseline --command-mode latest --gnss-quality rtk_fixed \
  --res 1280x720 --resizable --semantic-lidar-labels
```

새 B가 실행 중일 때 터미널 D:

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  prepare --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1
```

터미널 C는 반드시 새로 실행한다. 이번 버전은 ROS 시계0과 무관한 고유 session
로그 루트를 만들어 실험 노드의 `run_0` 파일 덮어쓰기를 막는다. D가 실제 노드의
로그 경로·제어 파라미터를 읽어 metadata에 연결하므로, 이전 C는 계측 preflight에서
거절될 수 있다. C:

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh ros2 launch launch_pkg simulation_experiment.launch.py \
  arrival_radius_m:=1.0 navigation_variant:=rejoin
```

준비 완료 뒤 터미널 D에서 딱1회 실행한다. **F9/F10은 누르지 않는다.**

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  run --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1 --profile baseline --arrival-radius-m 1.0 \
  --navigation-variant rejoin --command-mode latest --trace-paths
```

이번 폴더에는 기존 `metadata.json`, `frames.csv`, `summary.json`에 다음이 추가된다.

| 파일 | 내용 |
|---|---|
| `path_trace.jsonl` | 후보/컨트롤러 경로 원본·ID·goal 상태·TF·LiDAR 여유 이벤트 |
| `path_tracking.csv` | 실제 궤적과 그 시점 컨트롤러 경로의 횡/방향 오차·동일 OBB 여유 |
| `path_trace_summary.json` | 근접 구간 연결률, 최소 실제 여유 시점의 비교, 누락 사유 |

전체 geometry 분석은 기존 정지 확인 뒤 수행한다. 진행 중에는 관측만 하며
truth/장애물 OBB를 제어 입력으로 보내지 않는다. 경로 투영은 현재 pruned 경로의
처음3m만 사용하고 미래 수신 경로·취소된 목표·오래된 경로·최신 TF 대체는 사용하지
않는다. 분기 모호성/TF 누락 등은 명시적으로 보류한다.

끝의 JSON에 새 `path_trace`가 포함된다. 장애물 간격3m 이하이며 속도0.2m/s 이상인
근접 표본의 경로 연결이100%이고 최소 간격 시점도 연결되면 `association_verdict`가
`supported`다. v2 전체 `verdict`에는 아래의 동일 진행 위치 비교 조건도 요구한다.
`partial/unsupported`는 원인 분석에 필요한 관측이 부족하다는
뜻이지, 기존 주행 판정을 바꾸거나 추종 불량을 확정한 것이 아니다.
`supported` 역시 조향 적용/근본 원인 증명이나 별도의 안전 합격을 뜻하지 않는다.

**마지막 전체 JSON(`path_trace` 포함)과 `Trial output:` 경로를 전달한다.**
샘플 누락·제동 지연 등 계측 부하부터 확인하고, 물리 OBB가 같은 시각/장애물인
계획·실행 여유를 비교한다. LiDAR의 미래 구간 최소 여유와는 직접 같은 값으로
비교하지 않는다. 이후에만 원인에 따라 cost_penalty 또는 lookahead 중 하나의
변경 시험을 선택한다. 지금 legacy/latest 반복이나 두 파라미터 동시 튜닝은 하지 않는다.

### 2026-10-20 계측 v2: 목표 전환과 경로 경계 투영 구분

주행 파라미터·센서·안전 계층·기존 success/pass/aligned_stop 판정은 바꾸지 않았다.
이번 변경은 기록과 정지 후 분석뿐이다. 원본 `frames.csv` 열 계약도 유지한다.

1. `goal_context`를 추가했다. 복수 목표가 실행 상태인 전환
   (`overlapping_executing_goals`), 현재 목표와 연결된 새 경로 미확인
   (`active_goal_without_linked_window`), 목표 완료 후 여전히 이동 중
   (`succeeded_goal_moving`) 등을 구분한다. 취소 중/취소/중단 상태도 별도 표시한다.
   `controller_goal_status`, `active_goal_count`, 소비 속도 명령과 실제 제동,
   표본 간 속도 변화율도 기록한다. "목표 완료 후 이동"을 정상 감속이나 통신 장애라고
   자동 판정하지 않는다. 전환 표본을 근접 표본 분모에서 빼지 않고, 미래 상태를
   이용해 과거 UUID를 채우거나 종료된 경로를 정상 추종으로 되살리지 않는다.
2. `projection_region`과 `projection_longitudinal_offset_m`를 추가했다.
   차량의 수직 투영점이 pruned 경로 밖에 있으면 시작/끝/3m prefix/꼭짓점에 걸린
   비교로 명시한다. 이때 `projected_planned_clearance_m`와
   `projected_clearance_difference_m`는 기하 관측값으로 남기되,
   `same_station_planned_clearance_m`/`same_station_clearance_loss_m`는 비워 둔다.
   빈값은0이나 충돌이 아니다. 정확히 수직으로 대응하는 경계점은 비교 가능하다.
   내부 투영이어도 간격 차이는 위치와 차체 방향 차이가 합쳐진 값이지 순수 횡추종
   손실이나 제어 결함의 인과 증거가 아니다.

`critical_coverage`는 기존의 경로 연결률이다. 별도
`same_station_critical_coverage`는 **전체 근접 표본 중** 같은 진행 위치 비교가 가능한
비율이다. `association_verdict`, `same_station_verdict`를 분리하며 두 조건이 모두
100%이고 각각 최소 실제 간격 시점도 지원돼야 전체 `verdict=supported`다.
보류 표본을 지우거나 기준을 완화해100%로 만들지 않는다. `partial`이어도 표본별로
지원된 사실은 확인할 수 있으나 누락 구간까지 원인 분석이 완성됐다는 뜻은 아니다.

현재 로그는 재주행 없이 재분석할 수 있다. B/C/DDS/CARLA 연결이나 새로운 TF 조회를
사용하지 않고 저장된 경로·상태·당시 TF 변환 결과만 읽는다. 원본 v1/v2 이벤트를
지원하며 run_id 혼합/중복·역행 sequence/비정상 좌표는 거절한다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  trace-report \
  --trial ~/terrain_nav_data/experiments/runs/run_20261020_154142_150a2824
```

매번 원본 시험 폴더 아래 고유 `path_reanalysis_날짜_시간_ID/`를 생성한다.
기존 `metadata.json`, `frames.csv`, `summary.json`, `path_trace.jsonl`은 덮어쓰지 않는다.

```text
path_reanalysis_.../
  path_tracking.csv
  path_trace_summary.json
  analysis_metadata.json   # 원본 파일/분석 코드 SHA256, 분석 버전, 출처
```

이번 저장 로그를 v2로 분석하면 근접479개/경로 연결463개(96.66%)는 그대로다.
같은 진행 위치 비교는252개(52.61%)이고211개는 경로 시작점에 걸린 투영이다.
최소 실제 간격57.84cm는 보존하지만 그 시점의 진행 방향 위치차가-15.46cm이므로
기존11.48cm 차이를 동일 진행 위치의 추종 손실로 승인하지 않는다.
이 비율 감소는 새 데이터 손실이 아니라 이전에 섞여 있던 비교를 엄격히 분리한 결과다.
기존 주행 합격은 바꾸지 않으며, 이 수정 검증을 위해 S1 반복을 요구하지 않는다.

### S2 / S3 및 판정

`prepare`와 `run`의 `--scenario`를 함께 변경한다.

- S2: 중앙과 왼쪽 허용 통로를 장애물 차량 열로 막고 오른쪽 우회를 시험한다.
- S3: 허용 통로 전체를 가로막는다. 통로는 시작 직선 기준 기본±7m이며,
  차량 바운딩박스가 통로 밖으로 나가면 실패다. CARLA 세계 전체가 막힌 것은 아니다.
  밖으로 돌아나간 결과를 정상 회피로 세지 않는다. 설치면/실제 배치를 확인해야 한다.

S1/S2 합격은 충돌·바운딩박스 겹침·통로 이탈 없이 참위치 기준 목표3m 안에서
정지하고, 장애물 뒤에서 좌우0.5m/방향5도 이내를2초 유지하는 것을 요구한다.
이 기준은 현재 Lincoln 시험용이며 실차 합격 기준이 아니다.

S3는 최소8m 전진한 뒤 장애물/경로 차단으로 설명 가능한 정지를10초 유지해야 한다.
처음부터 움직이지 않거나 센서/명령이 끊겨 정지한 경우는 정상 차단 정지로 세지 않는다.
최대120초 **시뮬레이션 시간** 또는900초 벽시계 시간 후 종료한다.
`run`에서 Ctrl+C를 누르면 중단 기록을 남기고 클라이언트에 정지를 요청한다.
정지 확인이 없다는 경고가 나오면 Pygame 클라이언트도 종료한다.

최소 간격은 **실제 궤적의 회전된 바운딩박스 간격**이며, 필터된 점군이나 계획 경로의
간격이 아니다. 그러나 정확한 차체 mesh 간격도 아니므로 충돌 이벤트와 별도로 해석한다.
TF 불일치, 기록 간격0.5초 초과, 오래된 LiDAR/브리지 상태는 유효한 성공으로 승인하지 않는다.
경로 게이트 상태는 변경 시에만 보내는 유지형 이벤트다. 마지막 상태 변경 후 경과 시간은
통신 끊김 판정에 사용하지 않으며, 주행 중 상태 수신 여부와 주기적인 센서·최종명령·안전
상태의 수신 간격을 확인한다. `recording_issues`에 기록 무효 사유와 표본 수를 남긴다.

### 결과 확인

매 시도마다 아래 새 폴더를 만든다. 기존 ICCE/F9/F10 평가 레지스트리는 변경하지 않는다.

```text
~/terrain_nav_data/experiments/runs/run_YYYYMMDD_HHMMSS_ID/
  metadata.json    # 고정 배치·profile·클라이언트 해시·종료 원인
  frames.csv       # 참위치/제어용 위치·복귀 오차·조향·안전 상태
  summary.json     # 회피 성공 / 정상 차단 정지 / 실패 / 기록 유효성
```

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment report
```

새 `report_*.csv`에는 실패/중단도 포함한다. 다른 anchor나 클라이언트 소스 해시는
별도 그룹으로 표시한다. 내비게이션 소스·설정 해시도 기록해 수정 전후를 구분한다.
성공 시도만 골라 다시 실행해 대체하지 않는다.

개발 검사는 기존 회귀+새 계약 테스트, 패키지 빌드, localhost DDS의 TF·위치 기록
검사를 포함한다. **실제 CARLA의 S0/S1 주행 성공은 실행 결과로 별도 확인해야 한다.**
