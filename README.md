# GNSS LiDAR Outdoor Navigation Simulation

CARLA에서 GNSS 웨이포인트와 3D LiDAR를 사용해 주행·장애물 회피·경로 복귀를 검증하는 ROS 2 작업공간입니다.
실차 개발은 독립된 [실차 저장소](https://github.com/yunss01/gnss_lidar_outdoor_navigation_real)에서 진행합니다.

## 현재 상태

- GNSS 웨이포인트 주행과 F9 열린 경로·F10 폐회로 지원
- Nav2 경로 계획, LiDAR 장애물 지도, 독립 안전 게이트
- 별도 직선 실험에서 무장애물 S0와 단일 장애물 S1의 회피·복귀·도착 정지 비교
- 경로 복귀 방식과 최신 명령 수신 방식을 비교했으며, 현재는 목표 전환·경로 경계 구간의 추적 진단을 분석하는 단계

직선 실험의 위치는 CARLA 참위치 기반 합성 입력입니다. 실제 GNSS/IMU 융합이나 실차 조향·제동을 검증한 시스템은 아닙니다.

## 시작하기

Ubuntu 22.04, ROS 2 Humble, CARLA 0.10.0 환경을 사용합니다.
새 작업공간을 만드는 경우:

```bash
git clone https://github.com/yunss01/gnss_lidar_outdoor_navigation_sim.git ~/terrain_nav_ws
cd ~/terrain_nav_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

기존 작업공간은 로컬 변경사항을 확인한 뒤 갱신합니다.
직선 실험 실행기는 ROS와 이 작업공간을 자동으로 불러옵니다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment --help
```

서버·클라이언트·Nav2·시험 실행 명령과 반복 절차는
[실험 실행 안내](docs/SIMULATION_EXPERIMENT_GUIDE.md)에 정리되어 있습니다.
직선 실험에서는 F9/F10 대신 `prepare`와 `run` 명령을 사용합니다.

## 실험 조건과 결과

현재 직선 비교에서 사용하는 조합은 `profile=baseline`, `navigation_variant=rejoin`,
`command_mode=latest`, 도착 반경 `1.0m`입니다. 프로그램 기본값과는 구분하며 실행할 때 명시합니다.

- 일반 시뮬레이션은 ROS 도메인 **112**, 직선 실험은 **113**, 실차는 **42**입니다.
- CARLA 서버는 공유하므로 기존 클라이언트·Nav2와 직선 실험을 동시에 실행하지 않습니다.
- 결과는 `~/terrain_nav_data/experiments/`에 저장하며 매 시도와 실패 기록을 보존합니다.
- `success`, `aligned_stop_success`, `metrics_valid`, 경로 추적의 `partial`은 서로 다른 판정입니다.

계측 원본이 있으면 [저장 기록 재분석](docs/SIMULATION_EXPERIMENT_GUIDE.md#저장된-경로-기록-재분석)으로 재주행 없이 분석할 수 있습니다.
변화된 노면과 오차 조건에 대한 강건성, 실제 센서 기반 위치추정은 추가 검증 단계입니다.

## 주요 구성

- [`terrain_navigation_pkg`](src/terrain_navigation_pkg): GNSS 미션, Nav2 연결, 안전·실험·진단
- [`config_pkg`](src/config_pkg): 주행·Nav2 설정과 YAML 미션 예제
- [`launch_pkg`](src/launch_pkg): 실행 파일과 RViz 설정

기존 하드웨어 인터페이스는 호환성을 위해 남아 있지만, 현재 실차 작업공간은 이를 overlay/import하지 않습니다.
현재 상태와 실행법은 이 문서와 안내 문서에서 관리하고, 과거 변경 과정은 Git 이력으로 확인합니다.
