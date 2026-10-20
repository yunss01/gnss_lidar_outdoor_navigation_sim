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
