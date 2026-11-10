# 직선 주행 실험 실행 안내

고정 A→B 경로에서 회피·복귀·도착 정지를 비교하는 현재 실행 절차입니다.
일반 F9/F10 주행과 별개로 `scripts/experiment.sh`를 사용합니다.
위치는 CARLA 참위치 기반 합성 입력이며 실제 GNSS/IMU 위치추정 검증은 아닙니다.

## 환경과 조건

- Ubuntu 22.04, ROS 2 Humble, CARLA 0.10.0을 사용합니다.
- 직선 실험은 ROS 도메인 113입니다. 일반 시뮬레이션 112와 실차 42의 설정을 바꾸지 않습니다.
- 기존 Pygame 클라이언트와 Nav2는 종료하고 시작합니다. 도메인이 달라도 CARLA 서버 포트 2000은 공유합니다.
- 같은 anchor가 차량·스폰 위치·방향·경로를 고정합니다. 저장된 맵과 다른 맵에서 진행하지 않습니다.
- 공통 YAML과 외부 `manual_control.py`는 수정하지 않고 실험용 설정을 사용합니다.

현재 비교 조합은 다음과 같습니다. 프로그램 기본값을 가정하지 말고 명령에 명시합니다.

| 조건 | 사용 값 | 맞춰야 할 위치 |
|---|---|---|
| 오차 주입 | `baseline` | B 클라이언트와 D 실행기 |
| 복귀 방식 | `rejoin` | C Nav2와 D 실행기 |
| 명령 수신 | `latest` | B 클라이언트와 D 실행기 |
| GNSS 도착 반경 | `1.0m` | C Nav2와 D 실행기 |

`profile=baseline`은 오차 주입 없음이라는 뜻이며 `navigation_variant=baseline`과 다른 조건입니다.
`rejoin`은 최종 목표 앞 기준선상의 가상 복귀 지점을 사용하고, `latest`는 최신 제어 명령을 소비합니다.

## 빌드와 서버

모든 명령은 `~/terrain_nav_ws`에서 실행합니다. 실행기가 ROS와 이 작업공간을 불러옵니다.
처음에는 README의 전체 빌드를 하고, 관련 패키지를 다시 빌드할 때는 다음을 사용합니다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh colcon build \
  --packages-select terrain_navigation_pkg launch_pkg --symlink-install
```

터미널 A에서 CARLA 서버를 실행합니다. 이미 켜져 있으면 중복 실행하지 않습니다.

```bash
bash ~/carla/0.10.0/CarlaUnreal.sh
```

## 처음 한 번만 anchor 만들기

기존 `straight_v1/anchor.json`이 있으면 이 절을 건너뜁니다.
새 장소는 별도 경로를 사용하며 기존 anchor를 덮어쓰지 않습니다.

터미널 B에서 초기 후보를 고릅니다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 scripts/experiment_manual.py \
  --auto-straight-start --profile baseline --command-mode latest \
  --gnss-quality rtk_fixed --res 1280x720 --resizable --semantic-lidar-labels
```

자동 선택은 장애물 없는 도로나 우회 공간까지 보장하지 않습니다.
Pygame에서 직선·주변 교통·좌우 공간을 확인한 뒤 터미널 D에서 저장합니다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  capture --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json
```

기본 경로 길이는 40m이며 S1 장애물 중심은 출발점 전방 20m입니다.
A/B GNSS 좌표는 안테나 위치, 실제 이동·복귀·목표 오차 평가는 차량 원점 기준입니다.

## S1 실행

아래는 기존 anchor를 사용하는 단일 장애물 S1 절차입니다.
환경을 처음 점검할 때는 D의 `prepare`와 `run` 양쪽을 `--scenario S0`로 바꿔 무장애물부터 확인합니다.
**직선 실험에서는 F9/F10을 누르지 않습니다.** D의 `run`이 준비 검사 후 출발시킵니다.

### 터미널 B 클라이언트

기존 B가 있으면 아래 반복 절차로 정리한 뒤 실행합니다.
처음 anchor를 막 만들었다면 그 초기 B도 종료하고 anchor로 다시 실행합니다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 scripts/experiment_manual.py \
  --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --profile baseline --command-mode latest --gnss-quality rtk_fixed \
  --res 1280x720 --resizable --semantic-lidar-labels
```

별도 spawn/filter 옵션을 붙이거나 차량을 수동 이동시키지 않습니다.

### 터미널 D 장애물 준비

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  prepare --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1
```

준비는 출발 명령이 아닙니다. 화면·점군에서 배치를 확인합니다.
B를 재시작한 경우 이전 `prepared.json`을 재사용하지 말고 다시 준비합니다.

### 터미널 C Nav2

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh ros2 launch launch_pkg simulation_experiment.launch.py \
  arrival_radius_m:=1.0 navigation_variant:=rejoin
```

RViz가 필요하면 별도 터미널에서 실행합니다.

```bash
cd ~/terrain_nav_ws
bash scripts/experiment.sh rviz2 \
  -d src/launch_pkg/rviz/terrain_navigation.rviz \
  --ros-args -p use_sim_time:=true
```

### 터미널 D 출발과 기록

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  run --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json \
  --scenario S1 --profile baseline --arrival-radius-m 1.0 \
  --navigation-variant rejoin --command-mode latest --trace-paths
```

D는 활성 노드·센서·TF·반경·복귀 방식·명령 모드를 확인합니다.
검사가 실패하면 불일치를 해결하고, F9나 파라미터 완화로 우회하지 않습니다.
`--trace-paths`는 경로 계측용 선택 옵션이며 현재는 주입 없는 `baseline` 프로필에서 사용합니다.

## 다음 시험으로 초기화

1. D의 종료 결과와 차량 정지를 확인합니다.
2. C를 종료합니다.
3. **기존 B를 켜둔 채** D에서 아래 정리를 실행합니다.
4. B를 종료하고 같은 anchor로 다시 켭니다.
5. D `prepare` → C Nav2 → D `run` 순서로 반복합니다.

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  cleanup --anchor ~/terrain_nav_data/experiments/straight_v1/anchor.json
```

정리는 이 실험이 만든 actor만 식별해 제거하며 기록 파일을 지우지 않습니다.
동기식 월드의 시계를 진행하는 B를 먼저 끄면 정리가 기다릴 수 있습니다.
정지·종료 확인 경고가 있으면 다음 시험을 진행하지 말고 해당 기록을 확인합니다.

조건을 비교할 때는 위 표의 대응 위치를 모두 바꾸고 나머지 조건을 유지합니다.
오차 프로필을 추가하는 비교에서는 B/D의 `--profile`을 맞추고 `--trace-paths`는 제외합니다.

## 결과 읽기

매 시도는 `~/terrain_nav_data/experiments/runs/run_.../`의 새 폴더에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `metadata.json` | 요청·활성 설정, anchor·코드 출처 |
| `frames.csv` | 주행·안전·제어 관측 시계열 |
| `summary.json` | 주행 판정과 정지·복귀 진단 |
| `path_trace.jsonl` | 선택한 경로·목표 상태·TF·여유 이벤트 |
| `path_tracking.csv`, `path_trace_summary.json` | 경로 연결과 투영·여유 비교 |

판정은 구분해서 읽습니다.

- `metrics_valid`: 기록으로 평가할 수 있는지 여부입니다.
- `success`: 기존 주행 기준입니다. 정지까지 정렬을 유지했다는 뜻으로 확대하지 않습니다.
- `aligned_stop_success`: S1/S2에서 주행 성공과 정지까지 정렬 유지를 함께 봅니다. S0의 `null`은 실패가 아닙니다.
- `arrival_radius_verified`: C의 GNSS 제어 반경과 D 요청이 일치하는지 확인합니다.
- `goal_arrived`는 기존 차량 참위치 기준 목표 3m 이내라는 뜻입니다. GNSS 제어 반경 1m와 혼동하지 않습니다.
- 경로 추적의 `partial/unsupported`는 해당 비교의 관측 부족이며 주행 실패나 제어 결함을 자동 확정하지 않습니다.

경로 연결률과 같은 진행 위치의 여유 비교 가능 비율도 별도입니다.
원본 후보 경로를 수락된 제어 경로로 간주하거나, 경로 밖에 걸린 투영을 동일 위치 추종 손실로 해석하지 않습니다.
모든 실패·부족자료를 보존하고 anchor·차량·센서·코드·설정이 같은 조건끼리 비교합니다.

전체 시도 보고서는 다음 명령으로 생성합니다.

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment report
```

## 저장된 경로 기록 재분석

경로 계측 원본이 있는 시험은 재주행 없이 분석할 수 있습니다.
`/절대경로/시험폴더`를 실제 `run_...` 폴더로 바꿉니다.

```bash
bash scripts/experiment.sh python3 -m terrain_navigation_pkg.simulation_experiment \
  trace-report --trial /절대경로/시험폴더
```

저장된 경로·상태·당시 TF를 읽고 새 `path_reanalysis_.../`에 결과와 출처 해시를 남깁니다.
원래 summary·CSV·JSONL은 덮어쓰지 않으며 새 ROS/TF 관측으로 과거 기록을 보충하지 않습니다.

## 기존 F9와 F10 경로

일반 경로 주행은 직선 실험과 섞지 않고 ROS 도메인 112에서 사용합니다.
호환되는 `manual_control.py --external-control` 클라이언트와 CARLA 서버가 필요합니다.
이 저장소는 외부 CARLA 클라이언트 원본을 자동으로 교체하지 않습니다.

```bash
cd ~/terrain_nav_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=112
ros2 launch launch_pkg terrain_navigation_nav2.launch.py drive_enabled:=true
```

YAML 미션은 [경로 예제](../src/config_pkg/config/routes/mission_route_template.yaml)를 복사해
`mission_route_enabled`, `mission_route_file`, `mission_route_start` 인수로 지정합니다.
이 경로의 실차 인터페이스 소스가 남아 있다는 사실은 현재 독립 실차 시스템의 검증을 뜻하지 않습니다.
