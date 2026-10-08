# Traversability evidence v2

## 목적

v1은 각 셀을 `free / obstacle / unknown` 중 하나로 직접 분류한다. 이 구조는
장애물 확률이 낮으면 자유공간 확률이 자동으로 높아지는 이진 softmax를
사용하므로, 점이 적은 낮은 물체를 모르는 경우에도 자유공간으로 확신할 수
있다. motorhelmet 통제 실험에서 드러난 문제는 특정 물체 예외가 아니라 이
표현 자체의 문제다.

v2는 다음 두 양성 근거를 독립적으로 학습한다.

- `obstacle evidence`: 차량이 통과하면 안 되는 물체 또는 불연속의 관측 근거
- `passable-surface evidence`: 차량이 통과할 수 있는 표면을 실제로 관측한 근거

두 근거가 모두 부족한 셀은 `unknown`이다. **장애물 확률이 낮다는 이유만으로
자유공간을 만들지 않는다.** 서로 충돌하면 obstacle이 우선한다.

## 안전 불변조건

추론 결과가 passable이 되려면 아래 조건을 모두 만족해야 한다.

1. 셀이 실제로 관측되었다.
2. passable-surface 확률이 기준 이상이다.
3. obstacle 확률이 충분히 낮다.
4. 국소 지면 또는 visibility 같은 양성 support가 있다.
5. 불확실성이 기준 이하이다.
6. 같은 셀에 승인된 obstacle evidence가 없다.

하나라도 빠지면 obstacle 또는 unknown으로 남는다. 이 규칙은
`build_conservative_evidence_decision()`과 단위 테스트로 고정되어 있다.

## 차량 정책과 semantic label의 관계

CARLA semantic tag는 기본 supervision이다. 그러나 `Static`이라는 분류는
물체가 Lincoln MKZ로 통과 가능한지를 뜻하지 않는다. 통제 실험 물체에는
세션 `metadata.json`에 다음처럼 차량 정책을 기록한다.

```json
{
  "controlled_traversability_actors": [
    {
      "actor_id": 167,
      "blueprint": "static.prop.motorhelmet",
      "disposition": "obstacle",
      "policy_source": "vehicle_clearance_policy",
      "bbox_extent_xyz_m": [0.15, 0.12, 0.10]
    }
  ]
}
```

`disposition`은 다음 셋만 허용한다.

- `obstacle`: 이 차량이 통과하면 안 됨
- `passable`: 이 차량 정책상 통과 가능
- `ambiguous`: 아직 판정을 target으로 사용하지 않음

이 값은 해당 `object_idx`의 semantic 기본 분류보다 우선한다. 단, 같은 BEV
셀에 다른 obstacle 물체가 있으면 obstacle이 최종 우선한다. 높이나 blueprint
이름만 보고 disposition을 자동 추측하지 않는다. 차량 최저 지상고, 타이어,
안전 여유와 실제 시험으로 정한 정책을 metadata에 명시해야 한다.

한 개의 통제 actor를 정지 상태로 수집할 때는 recorder가 이 metadata를
자동으로 기록하게 다음처럼 실행한다. 정지 CARLA 장면의 scan은 반복 프레임이
완전히 동일할 수 있으므로 기본 수집값은 5 Hz, 3초다. actor ID와 blueprint는
그 실험에서 실제로 소환한 값으로 바꾼다. 현재 CARLA/ROS 구성에는 `/clock`이
없으므로 `use_sim_time:=false`를 사용한다.

```bash
RECORDER_ARGS=(
  --ros-args
  -p use_sim_time:=false
  -p enabled:=true
  -p record_rate_hz:=5.0
  -p save_sample_files:=true
  -p save_raw_points:=true
  -p save_semantic_labels:=true
  -p perception_capture_on_start:=true
  -p perception_capture_duration_s:=3.0
  -p perception_capture_output_directory:=/home/sukja/terrain_nav_data/learning/raw
  -p perception_capture_label:=controlled_motorhelmet_r10p00_center
  -p controlled_traversability_actor_id:=41
  -p controlled_traversability_disposition:=obstacle
  -p controlled_traversability_blueprint:=static.prop.motorhelmet
  -p controlled_traversability_policy_source:=vehicle_clearance_policy
)

ros2 run terrain_navigation_pkg \
  navigation_learning_recorder_node "${RECORDER_ARGS[@]}"
```

actor ID를 설정한 상태에서 semantic 저장을 끄면 recorder는 시작 단계에서
오류를 내어 잘못된 세션 생성을 막는다. 일반 수집에서는 기본값 `-1`을 그대로
사용한다. geometric LiDAR, odometry, semantic LiDAR가 모두 최신이고 semantic
scan timestamp가 geometric scan과 허용 오차 이내일 때만 3초 타이머가 시작된다.
종료 시 저장된 표본이 0개면 더 이상 `completed`로 기록하지 않고
`perception_capture_failed_no_samples`로 실패시킨다.

## 입력 표현

원래 4채널 BEV는 그대로 보존한다.

1. occupancy
2. log density
3. maximum height
4. same-cell vertical span

v2는 두 반경(기본 0.75 m, 1.50 m)마다 다음 채널을 추가한다.

- 주변 occupied 셀의 낮은 높이 분위수로 추정한 local ground 대비 max height
- local ground support confidence

따라서 기본 `lidar_evidence_bev`는 8채널이다. 이 특징은 물체 한 개에 대한
예외가 아니라, LiDAR 점이 셀 경계를 나누어 가져 same-cell span이 0이 되는
모든 낮은 돌출물에 적용된다. support가 부족하면 값을 억지로 추정하지 않고
invalid/unknown으로 남긴다.

## v2 데이터 생성

소스 빌드 후 실행한다.

```bash
cd /home/sukja/terrain_nav_ws
colcon build --packages-select terrain_navigation_pkg --symlink-install
source install/setup.bash

ros2 run terrain_navigation_pkg build_traversability_evidence_dataset
```

직접 Python 모듈로 실행할 수도 있다.

```bash
PYTHONPATH=/home/sukja/terrain_nav_ws/src/terrain_navigation_pkg \
python3 -m terrain_navigation_pkg.build_traversability_evidence_dataset \
  --raw-root /home/sukja/terrain_nav_data/learning/raw \
  --output-directory /home/sukja/terrain_nav_data/learning/traversability/v2
```

일부 세션만 별도의 pilot으로 만들 때는 `--session`을 반복한다. raw와 기존 v1,
v2는 건드리지 않도록 새 output 경로를 사용한다.

```bash
PYTHONPATH=/home/sukja/terrain_nav_ws/src/terrain_navigation_pkg \
python3 -m terrain_navigation_pkg.build_traversability_evidence_dataset \
  --raw-root /home/sukja/terrain_nav_data/learning/raw \
  --output-directory \
    /home/sukja/terrain_nav_data/learning/traversability/v2_motorhelmet_pilot_20260918 \
  --session session_20260918_125727_114556 \
  --session session_20260918_130311_055904 \
  --session session_20260918_131047_079774 \
  --session session_20260918_131351_894281 \
  --session session_20260918_131716_664846 \
  --session session_20260918_131920_097670 \
  --session session_20260918_132045_178612 \
  --session session_20260918_132221_401144 \
  --session session_20260918_132408_808475
```

기본적으로 각 세션 안에서 model input과 privileged target source가 완전히 같은
scan은 SHA-256 fingerprint로 하나만 남긴다. 중복은 `manifest.csv`에
`duplicate_identical_scan`과 원본 sample ID를 기록하므로 삭제가 아니라
derived dataset의 가중치 중복 방지다. 서로 다른 세션은 actor 정책이 다를 수
있어 같은 fingerprint여도 합치지 않는다. 빈 세션은 `summary.json`의
`empty_sessions`에 명시한다. 재현 실험 외에는
`--no-deduplicate-identical-scans`를 사용하지 않는다.

기본 출력은 아래에만 생성된다.

```text
/home/sukja/terrain_nav_data/learning/traversability/v2/
├── manifest.csv
├── summary.json
└── samples/<source_session>/sample_XXXXXX.npz
```

빌더는 raw 폴더 내부를 출력 위치로 지정하는 것을 거부하며, 기존 v1 기본
폴더를 출력 위치로 지정해도 거부한다.

## 주요 NPZ 배열

- `lidar_bev`: 원래 4채널 기하 BEV
- `lidar_evidence_bev`: 원래 BEV + multi-scale local-ground 특징
- `target_passable_surface_mask`: 양성 통과 가능 표면 근거
- `target_obstacle_evidence_mask`: 양성 장애물 근거
- `target_ambiguous_observed_mask`: 관측했지만 어느 양성 target도 아님
- `target_observed_mask`: semantic return이 존재한 셀
- `target_obstacle_instance_id`: 장애물 target의 CARLA object ID
- `target_controlled_*_point_count`: 통제 actor 정책이 만든 근거 수
- `local_ground_*`: 추정 높이, 상대 높이, support, valid mask
- `visibility_*`: 누적 지도에서 별도로 사용할 보수적 ray evidence

semantic tag와 object ID 원본은 privileged supervision이며 모델 입력 NPZ에는
복제하지 않는다. 대신 actor 정책, instance target, 집계치를 남겨 감사할 수
있게 한다.

## v2 감사와 시각화

학습 전에 모든 written sample의 불변조건을 검사한다. motorhelmet pilot에는
같은 차량 pose에서 actor만 제거한 세션을 대조군으로 지정한다.

```bash
PYTHONPATH=/home/sukja/terrain_nav_ws/src/terrain_navigation_pkg \
python3 -m terrain_navigation_pkg.visualize_traversability_evidence_dataset \
  --manifest \
    /home/sukja/terrain_nav_data/learning/traversability/v2_motorhelmet_pilot_20260918/manifest.csv \
  --output-directory \
    /home/sukja/terrain_nav_data/learning/traversability/visualizations/v2_motorhelmet_pilot_20260918 \
  --control-session session_20260918_130311_055904
```

감사는 다음을 확인하며 하나라도 위반하면 nonzero exception으로 중단한다.

1. passable, obstacle, ambiguous target 사이에 금지된 중첩이 없음
2. controlled actor 집계 셀이 disposition에 맞는 최종 target에 포함됨
3. manifest return 수와 derived NPZ return 수가 같음
4. obstacle actor ID가 instance target에 남아 있음
5. 각 controlled obstacle 셀이 actor-absent 대조군에서 이미 obstacle이 아님

출력은 `audit.csv`, `summary.json`, 300-DPI PNG contact sheet와 PDF다. 전체 BEV와
함께 actor 주변 21×21셀 확대 화면을 만들고, controlled 셀은 흰 테두리로
표시한다. 대조군 검사는 배경 장애물을 actor 증거로 오인하는 데이터 누수를
막는다.

2026-09-18 motorhelmet pilot 결과는 다음과 같다.

- 선택 세션 10개 중 1개는 0표본 실패 세션으로 보고됨
- raw 후보 160개 중 세션 내부 완전 중복 151개를 제외하고 9개 unique scan 생성
- controlled sample 8개, actor return 18개, controlled obstacle cell 13개
- 일반 불변조건 9/9 통과
- 대조군 검사 8/8 통과, 대조군에서 이미 obstacle인 controlled cell 0개
- 10m 정면 actor 34의 `(80,79)`, `(80,80)` 두 셀은 positive에서만 obstacle이며
  actor-absent 대조군에서는 obstacle이 아님

이 결과는 label 생성과 provenance가 맞다는 뜻이지, 모델 성능이 검증됐다는
뜻은 아니다.

같은 날 별도 scene에서 생성한 briefcase pilot 결과는 다음과 같다.

- dataset:
  `/home/sukja/terrain_nav_data/learning/traversability/v2_briefcase_pilot_20260918`
- audit:
  `/home/sukja/terrain_nav_data/learning/traversability/visualizations/v2_briefcase_pilot_20260918`
- actor-absent control 1개와 controlled session 9개
- raw 후보 128개 중 세션 내부 완전 중복 118개를 제외하고 10개 unique scan 생성
- 일반 불변조건 10/10, matched-control 검사 9/9 통과
- controlled return 89개, controlled obstacle cell 24개
- 대조군에서 이미 obstacle인 controlled cell 0개
- 10 m edge-on은 2 returns/1셀, broadside는 10 returns/2셀로 자세에 따른
  관측 밀도 차이를 분리함
- broadside 거리 sweep은 6/8/10/12 m에서 21/10/10/6 returns
- 좌우 0.5 m 및 lateral/forward 0.125 m 이동으로 인접 셀 경계 사례 확보

motorhelmet과 briefcase를 합하면 actor-absent control 2개, controlled unique
sample 17개, controlled return 107개, controlled obstacle cell 37개다. 이는
두 물체 family의 label 파이프라인 검증에는 유효하지만, 아직 모델 학습 성능을
대표할 만큼 scene/family가 다양하지 않다.

같은 날 scene03의 동일한 도로 배경과 차량 자세에서 motorhelmet, briefcase,
plasticchair를 교차 배치한 결과는 다음과 같다.

- dataset:
  `/home/sukja/terrain_nav_data/learning/traversability/v2_scene03_cross_family_pilot_20260918`
- audit:
  `/home/sukja/terrain_nav_data/learning/traversability/visualizations/v2_scene03_cross_family_pilot_20260918`
- actor-absent control 1개와 controlled session 10개
- raw 후보 133개 중 세션 내부 완전 중복 122개를 제외하고 11개 unique scan 생성
- 일반 불변조건 11/11, matched-control 검사 10/10 통과
- controlled return 96개, controlled obstacle cell 44개
- 대조군에서 이미 obstacle인 controlled cell 0개
- 10 m 중앙/오른쪽 1.2 m/왼쪽 1.2 m에서 motorhelmet은 2/2/2 returns,
  briefcase는 10/8/8 returns, plasticchair는 16/19/18 returns다.
- plasticchair 중앙 yaw 90도는 11 returns로 yaw 0도의 16 returns와 자세에
  따른 관측 밀도 및 셀 형태 차이를 제공한다.
- contact sheet를 직접 확인한 결과 모든 controlled evidence가 최종 obstacle
  target 및 actor instance provenance와 일치했다.

세 pilot의 누적치는 actor-absent control 3개, controlled unique sample 27개,
controlled return 203개, controlled obstacle cell 81개다. scene03 교차 설계로
family와 scene가 일대일로 묶이는 교란을 일부 해소했지만, 모델 성능을 주장하기
전에는 각 family를 둘 이상의 추가 scene에서 교차 수집하고 scene/family 단위의
고정 split을 만들어야 한다.

scene04에서는 같은 차량 자세에서 actor-absent control과 세 family의 10 m 중앙
조건만 수집했다.

- dataset:
  `/home/sukja/terrain_nav_data/learning/traversability/v2_scene04_cross_family_pilot_20260918`
- audit:
  `/home/sukja/terrain_nav_data/learning/traversability/visualizations/v2_scene04_cross_family_pilot_20260918`
- raw 후보 46개 중 세션 내부 완전 중복 42개를 제외하고 4개 unique scan 생성
- 일반 불변조건 4/4, matched-control 검사 3/3 통과
- controlled return 27개, controlled obstacle cell 13개
- 대조군에서 이미 obstacle인 controlled cell 0개
- motorhelmet/briefcase/plasticchair의 return은 각각 2/10/15개이며 scene03의
  중앙 조건 2/10/16개와 거의 동일하게 재현됐다.

네 pilot의 누적치는 actor-absent control 4개, controlled unique sample 30개,
controlled return 230개, controlled obstacle cell 94개다. background coverage는
motorhelmet scene01/03/04, briefcase scene02/03/04, plasticchair scene03/04다.
scene04는 scene-held-out 검증 후보로 고정하고 추가 pose를 같은 검증 장면에
계속 넣지 않는다.

scene05는 왼쪽 근거리 연석·나무·가로등·건물과 오른쪽의 열린 도로가 만드는
비대칭 배경이며, 고정 scene test 후보로 수집했다. 기존 세 family 외에 학습에
포함하지 않을 trashcan family를 추가했다.

- dataset:
  `/home/sukja/terrain_nav_data/learning/traversability/v2_scene05_family_holdout_test_20260918`
- audit:
  `/home/sukja/terrain_nav_data/learning/traversability/visualizations/v2_scene05_family_holdout_test_20260918`
- raw 후보 62개 중 세션 내부 완전 중복 57개를 제외하고 5개 unique scan 생성
- 일반 불변조건 5/5, matched-control 검사 4/4 통과
- controlled return 42개, controlled obstacle cell 14개
- 대조군에서 이미 obstacle인 controlled cell 0개
- motorhelmet/briefcase/plasticchair/trashcan01의 return은 2/10/16/14개,
  obstacle cell은 2/2/6/4개다.
- trashcan01은 폭 약 0.25 m, 관측 높이 약 0.90 m의 단단한 비통과 물체이며
  학습 split에서 제외하고 family-held-out 평가에만 사용한다.

전체 누적치는 actor-absent control 5개, controlled unique sample 34개,
controlled return 272개, controlled obstacle cell 108개다. scene04는 validation,
scene05는 test로 고정하고, trashcan은 family-held-out test로 유지한다.

## 현재 단계와 다음 단계

2026-09-18에 label 파이프라인 확인을 넘어, 장면 누수가 없는 첫 v2 학습·평가
파이프라인까지 만들었다. 그래도 결과는 **비배포 pilot**이며 실시간 candidate
filter에는 연결하지 않는다.

### 고정 데이터 명세와 split

- 결합 dataset:
  `/home/sukja/terrain_nav_data/learning/traversability/v2_controlled_dataset_20260918`
- 고정 experiment:
  `/home/sukja/terrain_nav_data/learning/traversability/v2_controlled_experiment_20260918`
- dataset fingerprint:
  `e2533b4b174cc206ec47da093dc856e621288deaa7222e43810317127e08ddb9`
- train: scene01--03, 30 samples, controlled obstacle 81셀
- validation: scene04, 4 samples, controlled obstacle 13셀
- test: scene05, 5 samples, controlled obstacle 14셀
- `trashcan01`은 test-only family holdout이며 통계가 아니라 단일 sentinel이다.
- 같은 scene의 control과 positive는 항상 같은 split에 있다. 자동 검사 결과
  scene leakage는 0이다.
- 결합 dataset 전수 감사 결과 39/39 PASS, 34 controlled samples,
  272 controlled returns, 108 controlled obstacle cells다.

`prepare_traversability_evidence_experiment`는 무작위 분할을 하지 않는다.
명시적인 `dataset_spec.json`과 실제 NPZ의 actor blueprint/disposition을 대조하고,
각 scene에 control이 정확히 하나인지, family holdout이 train/validation에
노출되지 않았는지, 각 controlled sample에 target cell이 존재하는지 검사한다.
manifest/spec/split의 SHA-256과 최종 fingerprint도 저장한다.

### v2 학습 계약

- 입력은 `lidar_evidence_bev` 8채널이다.
- passable과 obstacle은 softmax 상호배타 class가 아니라 독립 sigmoid head다.
- passable/obstacle 양성 근거가 없는 ambiguous/unobserved 셀에는 loss를 주지
  않는다. 낮은 obstacle 확률을 passable label로 바꾸지 않는다.
- family-balanced sampler와 controlled-instance-balanced auxiliary loss를 쓴다.
- 각 actor-present sample과 같은 scene의 actor-absent control을 함께 넣는다.
  controlled actor 셀에서 present obstacle logit이 control보다 margin만큼 높아야
  하며, control의 같은 셀 obstacle logit은 억제한다.
- trainer는 test row를 로드하지 않는다. checkpoint 선택은 scene04 validation만
  사용한다.

paired loss가 없는 첫 진단 모델은 validation obstacle IoU 0.811과 controlled
recall 13/13을 보였지만, motorhelmet/briefcase 위치의 actor-absent control
obstacle 확률도 약 0.9999였다. 즉 작은 actor를 본 것이 아니라 배경/위치 prior를
맞혀 recall이 부풀어 있었다. 이 모델은 실패 진단으로만 보존한다.

paired-counterfactual pilot은 scene04에서 다음을 보였다.

- obstacle IoU 0.780, precision 0.949, recall 0.814
- passable IoU 0.705, precision 0.961, recall 0.726
- controlled obstacle cell 13/13, instance 3/3 검출
- 세 family의 actor-present obstacle 확률 약 0.9997--1.0
- 같은 셀의 actor-absent control 확률 약 0.0009--0.0023
- family별 present-minus-control 평균 delta 0.997 이상

모델과 임계값을 `model_selection.json`에 먼저 고정한 뒤 scene05 test를 한 번만
열었다. test 결과는 다음과 같다.

- obstacle IoU 0.767, precision 0.934, recall 0.811
- passable IoU 0.727, precision 0.990, recall 0.732
- controlled obstacle cell 11/14, instance any-detection 3/4
- briefcase 2/2, plasticchair 5/6, trashcan01 4/4셀 검출
- **motorhelmet 0/2셀, instance miss**
- trashcan01 1개 성공은 unseen-family 일반화의 증명이 아니라 sentinel 통과다.

따라서 이 checkpoint는 배포 불가다. test miss를 보고 scene05에 맞춘 threshold나
학습 설정을 다시 고르지 않는다. 다음 dataset version에서는 scene05도 이미 본
test로 간주하고, 새로운 untouched test scene을 마련해야 한다.

다음 작업의 우선순위는 모델 크기나 임계값 조정이 아니라 데이터 설계다.

1. 1--2셀/1--3 return의 희소한 작은 비통과 장애물을 여러 새 scene, 거리,
   방위에서 actor-absent control과 쌍으로 수집한다.
2. 단단한 roadside family(낮은 볼라드, 작은 상자, 얇은 기둥 등)를 추가하되,
   family와 scene가 일대일로 결합되지 않게 교차 배치한다.
3. 새 scene 하나는 validation, 또 다른 하나는 untouched test로 완전히 보류한다.
4. paired delta, control false-positive, instance recall을 checkpoint 선택의 필수
   지표로 유지한다. cell IoU만으로 모델을 고르지 않는다.
5. scene/family별 표본 수가 충분해진 뒤 calibration과 uncertainty를 추가하고,
   그 다음에만 shadow mode 연결을 검토한다.

MLflow experiment `traversability_evidence_v2`에는 reference-only 정책으로
dataset milestone run `9155e39a0ec84b9a8a37eefd915684d6`와 비배포 model run
`ebdd812a389948aa94c5f947f67be960`을 기록했다. 모델 run에는 scene05
motorhelmet miss를 known failure로 명시했다. 기존 v1 모델과 obstacle candidate
노드는 비교 baseline으로 유지한다.

## 2026-09-21 실패 원인 분석과 개발 장면 교차검증

이미 소비된 scene05 test의 motorhelmet miss는 새 임계값을 고르는 데 사용하지
않고 원인 진단에만 사용했다. scene04 성공 사례와 raw evidence를 비교한 결과,
scene05에서도 motorhelmet은 2 returns/2셀로 관측됐고 local ground 대비 높이는
약 0.16 m였다. 즉 LiDAR 신호가 없었던 것이 아니다. 같은 물체와 거의 같은
기하 신호에 대해 scene04의 paired obstacle delta는 약 `+0.999`였지만 scene05는
`-0.046`이었다. 넓은 receptive field와 절대 좌표를 쓰는 U-Net obstacle head가
물체보다 배경 배치에 의존했다는 것이 가장 강한 설명이다.

이 가설을 물체별 예외 규칙으로 막지 않고 검증하기 위해 두 구조를 같은 조건
(`lr=0.001`, 최대 50 epoch, base channel 8)에서 scene01--04
leave-one-scene-out으로 비교했다. 각 fold manifest에는 기존 test row가 남지만
trainer/evaluator는 이를 로드하지 않는다.

| 개발 CV 지표 | local evidence | context U-Net |
| --- | ---: | ---: |
| controlled cell recall 평균 / 최저 | 0.935 / 0.846 | 0.941 / 0.846 |
| controlled instance recall 평균 / 최저 | 0.941 / 0.875 | 0.941 / 0.875 |
| obstacle IoU 평균 / 최저 | 0.726 / 0.582 | 0.781 / 0.595 |
| paired control obstacle 확률 평균 / 최대 | 0.026 / 0.076 | 0.076 / 0.146 |
| paired present-control delta 평균 / 최저 | 0.919 / 0.745 | 0.890 / 0.759 |
| passable IoU 평균 / 최저 | 0.400 / 0.000 | 0.451 / 0.007 |

결론은 다음과 같다.

1. 작은 통제 장애물 recall은 두 구조가 사실상 같다.
2. U-Net은 전체 obstacle segmentation이 더 좋지만, actor-absent control의 같은
   셀에서 obstacle 확률이 local model보다 약 3배 높다.
3. local model은 절대 좌표, pooling, 원거리 배경을 제거한 17×17셀 receptive
   field를 사용하므로 다음 obstacle-safety 후보로 유지한다.
4. 두 구조 모두 passable의 장면별 최저 성능이 0에 가까우므로 어느 checkpoint도
   배포하거나 `free` 생성기로 쓰지 않는다. 근거가 불충분하면 unknown이라는
   기존 안전 불변조건을 그대로 유지한다.
5. 이 비교는 이미 소비된 scene05 test를 다시 선택 지표로 사용하지 않았다.
   scene05는 known-failure 진단 자료일 뿐이다.

재현 가능한 결과는 다음 두 경로에 있다.

- local:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/cv_local_evidence_lr1e3_20260921`
- U-Net:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/cv_unet_context_lr1e3_20260921`

### 다음 데이터 버전의 수집 계약

다음 수집은 한 물체를 여러 위치에 반복하는 sweep보다 **배경과 family의 결합을
끊는 것**을 우선한다.

1. 새 개발 장면 scene06과 scene07을 서로 다른 배경으로 정한다. 각 장면에서
   동일 차량 pose의 actor-absent control을 먼저 저장한다.
2. 두 장면 모두 motorhelmet, briefcase, plasticchair, trashcan01을 교차 배치한다.
   한 family를 한 장면에만 두지 않는다.
3. 각 장면에서 1--3 return/1--2셀인 희소 조건을 최소 2개 포함한다. 거리는
   8--12 m, lateral 위치는 중앙과 좌/우 중 하나로 분산하되 모든 조합을 반복할
   필요는 없다.
4. passable 일반화를 위해 각 장면에 서로 다른 차량 pose의 actor-absent scan을
   하나 더 추가한다. 장애물 pair용 control과 섞지 않고 pose ID를 명시한다.
5. scene08은 새 untouched test로 봉인한다. target/metadata 감사는 허용하지만
   개발 중 모델 출력과 metric은 열지 않는다. 최종 구조·threshold·선택 기준을
   고정한 뒤 한 번만 평가한다.

다음 개발 checkpoint의 최소 gate는 새 데이터가 들어온 뒤 다시 확정한다. 현재
기준으로는 controlled instance recall 최저 0.95 이상, paired control obstacle
확률 최대 0.05 이하, paired delta 최저 0.80 이상을 동시에 요구한다. passable은
모든 개발 장면 IoU가 0.60 이상이 되기 전까지 navigation free-space source로
승격하지 않는다. 이 gate를 못 넘으면 임계값으로 덮지 않고 데이터/표현 문제로
되돌린다.

동일한 reference-only 정책으로 개발 CV도 MLflow에 기록했다.

- local evidence CV: `5be26a804fbf4bd69dba5fd89b91dfc4`
- context U-Net CV: `42a0aa35c29a4226b0d5560266ca2518`

두 run 모두 `deployable=false`, `frozen_test_accessed=false`이며 checkpoint나
dataset 파일을 artifact store에 복사하지 않고 원본 경로와 summary hash만
기록한다.

## 2026-09-21 scene06 context diagnostic

교차검증 결론을 독립적인 새 배경에서 반증하기 위해 scene06을 수집했다. 넓은
교차로지만 주차건물, 난간, 화분, 건물군이 비대칭 문맥을 만들며 actor 배치
영역 자체는 평탄하고 비어 있다.

- Pose A actor-absent matched control 1개
- motorhelmet 10 m 중앙/오른쪽 1.2 m/왼쪽 1.2 m 3개
- briefcase, plasticchair, trashcan01 10 m 중앙 각 1개
- 차량을 약 13 m 옮기고 약 160--180도 회전한 Pose B passable control 1개
- raw 97 frames에서 정지 중복 89개를 제외해 8 unique scan 생성
- controlled sample 6개, 38 returns, 15 obstacle cells
- audit 8/8 PASS, matched control에서 기존 obstacle인 controlled cell 0개

같은 물리 장면에서 두 차량 자세를 정직하게 표현하기 위해 `scene_id`와 별도로
`pair_group`을 split manifest에 추가했다. paired loss와 평가기는 같은
`scene_id/pair_group`의 control만 사용한다. 기존 manifest에는 pair_group이
없으므로 이전처럼 scene_id를 기본 pair group으로 사용한다.

scene01--04만 train, scene06만 validation으로 고정해 동일 설정의 U-Net과
bounded-context CNN을 다시 학습했다. 이미 소비된 scene05는 provenance를 위해
test row로 남겼지만 `consumed_test_do_not_use`로 표시했고 evaluator는 명시적인
override 없이는 해당 split 접근을 거부한다.

| scene06 validation | context U-Net | bounded-context CNN |
| --- | ---: | ---: |
| obstacle IoU | 0.909 | 0.717 |
| passable IoU | 0.544 | 0.000 |
| controlled cell recall | 12/15 | 12/15 |
| controlled instance recall | 4/6 | 6/6 |
| motorhelmet cell / instance recall | 2/5, 1/3 | 4/5, 3/3 |

bounded CNN은 motorhelmet instance는 모두 찾았지만 actor-absent 같은 셀의
obstacle 확률이 motorhelmet 평균 0.174, briefcase 0.238, trashcan01 0.183으로
높았고 passable을 한 셀도 승인하지 못했다. 따라서 bounded CNN을 U-Net 대신
선택한다는 가설은 기각한다.

세 motorhelmet 조건은 모두 control 대비 local-ground 상대 높이가 약
0.19--0.23 m 증가했다. 이 값은 사전에 정한 차량의 0.15 m 비통과 정책을
명백히 넘는다. 물체 family를 외우게 하지 않고 이 물리 정책을 보존하도록
다음 hard obstacle evidence를 추가했다.

1. 유효한 multi-scale local ground 대비 높이 0.15 m 이상
2. 같은 셀 vertical span 0.15 m 이상
3. 기존 sensor-frame absolute height -1.40 m 이상

hard obstacle은 learned obstacle과 OR로 결합하며 learned passable이 지울 수
없다. scene06 U-Net에 적용한 결과는 다음과 같다.

- obstacle IoU 0.913, precision 0.916, recall 0.997
- controlled cell 15/15, instance 6/6
- matched-control 같은 셀 false obstacle 0/15
- bounded CNN + hard evidence는 13/15셀, 6/6 instance, obstacle IoU 0.721

따라서 다음 개발 후보는 **U-Net learned evidence + vehicle-clearance hard
evidence + unknown 우선 결정**이다. 이는 motorhelmet 예외가 아니라 차량이
넘을 수 없는 돌출 높이라는 일반 불변조건이다. 다만 scene06 하나의 개발
validation 결과이고 passable IoU도 사전 gate 0.60에 못 미치므로 여전히
`deployable=false`다. scene07 새 배경에서 같은 계약을 다시 검증하기 전에는
실시간 navigation authority에 연결하지 않는다.

결과 경로:

- scene06 dataset:
  `/home/sukja/terrain_nav_data/learning/traversability/v2_scene06_context_diagnostic_20260921`
- frozen diagnostic experiment:
  `/home/sukja/terrain_nav_data/learning/traversability/v2_scene06_diagnostic_experiment_20260921`
- U-Net:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/scene06_diagnostic_unet_context_20260921`
- bounded CNN:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/scene06_diagnostic_bounded_context_20260921`

scene06 진단 milestone은 기존 파일을 복사하지 않는 reference-only 정책으로
MLflow experiment `traversability_evidence_v2`에도 기록했다.

- scene06 diagnostic dataset: `1fe4d4e285b84239901527be98ea165d`
- selected U-Net diagnostic: `6c573807c4c34a55b3e9c69b3434ea46`
- rejected bounded-context diagnostic: `01f81b0361f144828e83f1c8c4537693`

세 run 모두 `deployable=false`, `consumed_test_accessed=false`다. 모델 run에는
raw learned-head 지표와 고정된 vehicle-clearance fusion 지표를 분리해 기록했다.

## 2026-09-21 scene07 fixed-policy replication

scene06 결과를 본 뒤 정한 체크포인트, threshold, 0.15 m clearance 정책을 전혀
바꾸지 않고 새 scene07에서 재현 시험을 수행했다. scene07은 좁은 가로수 도로,
주차 차량, 길가 시설물로 구성돼 scene06의 넓은 교차로와 문맥이 다르다.

- Pose A matched control 1개
- motorhelmet 10 m 중앙/오른쪽 1.2 m/왼쪽 1.2 m 3개
- briefcase, plasticchair, trashcan01 10 m 중앙 각 1개
- 이동·회전한 Pose B passable control 1개
- raw 102 frames 중 정지 중복 94개를 제외해 8 unique scan 생성
- controlled sample 6개, 46 returns, 15 obstacle cells
- audit 8/8 PASS, matched control의 같은 target cell 기존 obstacle 0/15

scene06에서 선택한 epoch 21 U-Net을 재학습하지 않고 평가한 결과는 다음과 같다.

| scene07 fixed evaluation | raw U-Net | U-Net + clearance |
| --- | ---: | ---: |
| obstacle IoU | 0.858 | 0.911 |
| passable IoU | 0.613 | 동일 learned passable 사용 |
| controlled cell recall | 11/15 | 14/15 |
| controlled instance recall | 3/6 | 6/6 |
| controlled unsafe-passable cells | 3/15 | 1/15 |
| controlled unsafe-passable instances | 3/6 | 0/6 |
| paired-control false obstacle | 해당 없음 | 0/15 |

raw U-Net은 motorhelmet 4/4셀과 3/3 instance를 모두 놓쳤다. mean paired
control probability는 0.023으로 낮았지만 present-control delta도 0.046에
불과했다. 따라서 learned obstacle head의 작은 희소 장애물 문맥 의존성은
scene07에서도 재현됐다.

고정 clearance fusion은 6/6 instance를 모두 잡았다. 놓친 target 1셀은 왼쪽
helmet의 바닥 가장자리 1 return으로 local-ground 상대 높이가 0.035--0.050 m였다.
같은 instance의 인접 셀은 0.268--0.285 m여서 hard obstacle로 검출됐다. 이는
0.15 m 경계를 낮출 근거가 아니며 cell 단위 target과 차량 단위 안전 결정을
구분해야 하는 사례다.

scene07 passable IoU 0.613은 장면별 0.60 gate를 넘었지만 scene06은 0.544로
남아 있으므로 전체 gate는 아직 실패다. raw learned motorhelmet gate도 실패했다.
따라서 `deployable=false`와 navigation authority 미연결 상태를 유지한다.

다음 단계는 scene01--07 개발 자료만 사용한 scene-wise 교차검증이다. fusion
정책은 고정하며 scene별 성능을 확인한 뒤 하나의 checkpoint를 동결한다. 그 전까지
scene08은 수집하더라도 모델 출력을 열지 않는 untouched test로 유지한다.

결과 경로:

- scene07 dataset:
  `/home/sukja/terrain_nav_data/learning/traversability/v2_scene07_context_replication_20260921`
- fixed-policy evaluation:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/scene07_fixed_unet_clearance_replication_20260921.json`
- replication summary:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/scene07_fixed_policy_replication_20260921_summary.json`

MLflow reference-only runs:

- scene07 dataset: `c07828b545864f5f8cba3c1c0bf6db4a`
- fixed-policy replication with unsafe-passable metrics:
  `6d1c5674ded64b2ab7454f0ff68ed2c2`

초기 fixed-policy MLflow run `636979b777c74041a82f1c6757147f13`은 지표 정의
확장 전의 이력으로 보존했다. 현재 판단에는 위의 새 run을 사용한다.

## 2026-09-21 scene01--07 development cross-validation

scene01--07을 leave-one-scene-out 방식으로 다시 비교했다. 이미 결과를 열어 본
scene05는 더 이상 test라고 부르지 않고 명시적으로 development로 승격했다.
미래 scene08은 데이터에도 존재하지 않으며 접근하지 않았다.

공유 U-Net에서 passable/obstacle loss가 같은 표현을 서로 다른 방향으로 끌어당기는
문제를 검증하기 위해 `decoupled_unet_context`를 추가했다. 이 구조는 동일한 입력을
받지만 passable과 obstacle에 완전히 독립적인 1-output U-Net을 사용한다. 두 head는
parameter와 gradient를 공유하지 않는다. threshold와 0.15 m vehicle-clearance
fusion은 비교 중 바꾸지 않았다.

| 7-fold development metric | shared U-Net | decoupled U-Net |
| --- | ---: | ---: |
| raw obstacle IoU mean / min | 0.783 / 0.499 | 0.866 / 0.644 |
| raw passable IoU mean / min | 0.467 / 0.000 | 0.826 / 0.000 |
| raw controlled instance mean / min | 0.893 / 0.667 | 0.958 / 0.833 |
| fused controlled instance mean / min | 0.982 / 0.875 | 1.000 / 1.000 |
| fused unsafe-passable instance max | 0.000 | 0.000 |
| fused paired-control false obstacle max | 0.000 | 0.467 |
| paired-control probability max | 0.198 | 0.531 |
| paired present-control delta min | 0.422 | 0.307 |

분리형은 scene01--06의 passable IoU를 모두 0.948 이상으로 끌어올렸다. 따라서
shared representation의 task interference 가설은 지지된다. 그러나 scene07을
통째로 holdout한 fold에서는 passable IoU가 0이었고, actor-absent control의 target
cell 7/15를 obstacle로 오검출했다. 이 결과는 출력 분리만으로 새 배경 일반화가
해결되지 않으며, 현재 병목이 다양한 actor-absent hard negative의 부족이라는 것을
보여 준다.

`unsafe-passable instance`는 한 instance에서 obstacle로 받아들인 cell이 하나도
없으면서 controlled cell 중 하나 이상을 passable로 승인했을 때만 센다. 이 정의로
두 구조의 fused unsafe-passable instance는 전 fold 0이었다. 그러나 분리형의
scene07 control 오탐은 주행 가능 공간을 과도하게 막으므로 배포 gate 실패다.

결론은 다음과 같다.

- 배포 구조와 최종 checkpoint는 선택하지 않는다(`deployable=false`).
- `decoupled_unet_context`는 task interference를 줄인 development-only 후보로만
  유지한다.
- learned evidence와 clearance fusion은 아직 navigation authority에 연결하지 않는다.
- 다음 학습 전에는 새로운 도로 문맥과 여러 차량 pose의 actor-absent hard negative를
  추가한다. scene08은 구조·threshold·gate를 고정하기 전까지 untouched로 유지한다.

결과 경로:

- shared U-Net CV:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/cv_unet_context_scenes01_07_fixed_clearance_20260921`
- decoupled U-Net CV:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/cv_decoupled_unet_scenes01_07_fixed_clearance_20260921`
- selection record:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/scene01_07_architecture_selection_20260921.json`

MLflow reference-only runs:

- shared U-Net 7-fold CV: `eb3eb5706d5747a6b105b901c36bb9dc`
- decoupled U-Net 7-fold CV: `2712ceab497544458711f1419b43bb77`

## 2026-09-21 branch-independent training protocol

위 교차검증의 두 U-Net은 parameter와 gradient는 공유하지 않았지만 optimizer,
scheduler, early stopping, `best.pt` 선택을 결합 validation loss로 공유했다. 특히
scene07의 기존 `best.pt`는 epoch 2에서 선택되어 passable IoU가 0이었지만,
`last.pt`의 passable IoU는 0.670이었다. 따라서 기존 결과에는 표현의 실패뿐 아니라
학습 제어와 checkpoint 선택의 결합 오류가 섞여 있었다.

`decoupled_unet_context` 학습기를 다음과 같이 수정했다.

- passable/obstacle branch마다 별도 AdamW와 ReduceLROnPlateau를 사용한다.
- branch별 validation objective와 stale epoch를 사용해 독립적으로 조기 종료한다.
- passable branch는 validation passable loss가 가장 낮은 epoch를 선택한다.
- 선택한 passable branch를 고정한 뒤 저장된 모든 obstacle epoch를 다시 평가한다.
- obstacle epoch는 unsafe-passable instance 0, paired-control false obstacle 0,
  controlled-instance recall 0.95 이상, obstacle IoU 순서로 선택한다.
- `best_passable_branch.pt`, `best_obstacle_branch.pt`, 조립된 `best.pt`, 모든 후보의
  지표를 담은 `branch_selection.json`을 남긴다.

threshold, 0.15 m clearance 정책, 입력 데이터, fold seed는 바꾸지 않고 같은
scene01--07 leave-one-scene-out 평가를 다시 실행했다. scene08은 존재하지 않으며
접근하지 않았다.

| 7-fold development metric | 기존 결합 제어 | branch 독립 제어 |
| --- | ---: | ---: |
| raw passable IoU mean / min | 0.826 / 0.000 | 0.967 / 0.934 |
| raw obstacle IoU mean / min | 0.866 / 0.644 | 0.871 / 0.692 |
| raw controlled instance mean / min | 0.958 / 0.833 | 0.905 / 0.500 |
| fused controlled instance mean / min | 1.000 / 1.000 | 1.000 / 1.000 |
| fused paired-control false obstacle max | 0.467 | 0.000 |
| fused unsafe-passable instance max | 0.000 | 0.000 |

scene07 passable IoU는 0에서 0.962로 회복됐고 paired control false obstacle은
7/15에서 0/15로 줄었다. 따라서 이전 scene07 passable 붕괴와 control 오탐의
상당 부분은 결합 checkpoint 선택 오류였다.

반면 raw controlled-instance recall은 scene06 0.833, scene07 0.500에 그쳤다.
두 fold에서는 저장된 obstacle 후보 중 control 오탐 0과 recall 0.95를 동시에
만족하는 epoch가 없었다. 이는 문제가 단순히 actor-absent hard negative의
부족만은 아니며, 희박한 작은 장애물 신호와 배경 오탐 억제 사이의 실제 tradeoff가
남아 있음을 의미한다. 특히 scene07 motorhelmet은 3/3 instance를 raw head가
놓쳤지만 passable로 승인하지 않고 unknown으로 남겼고, 고정 clearance fusion은
6/6 controlled instance를 검출했다.

현재 결론은 다음과 같다.

- branch 독립 학습/선택을 이후 development protocol로 사용한다.
- passable gate와 fused safety gate는 통과했지만 raw small-obstacle gate는
  실패했으므로 `deployable=false`와 navigation authority 미연결을 유지한다.
- 새 threshold나 물체별 예외 규칙으로 recall을 복구하지 않는다.
- 다음 표현 실험은 ego-motion으로 보정한 3--5 scan temporal evidence이다.
- 현재 정지 수집 세션의 반복 frame은 동일 fingerprint이므로 temporal 실험에는
  움직이는 연속 기록이 필요하다. scene08은 계속 untouched로 둔다.

결과 경로:

- branch-independent CV:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/cv_decoupled_unet_independent_selection_scenes01_07_20260921`
- training-protocol selection record:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/scene01_07_training_protocol_selection_20260921.json`
- MLflow reference-only protocol run:
  `fd850a3089cf4600b762ffd94687307c`

## 2026-09-21 ego-motion temporal pilot

정지 상태에서 반복 저장된 동일 scan을 합치는 방식은 새 evidence를 만들지 못한다.
따라서 scene08을 건드리지 않고 scene09의 같은 시작 자세에서 저속 직진한 control과
motorhelmet obstacle을 각각 기록했다.

- control: `session_20260921_192933_420951`, 58 frames, 4.467m 이동,
  고유 scan 44개.
- obstacle: `session_20260921_193450_213974`, 60 frames, 3.501m 이동,
  고유 scan 37개.
- 시작 위치 오차 0.005m, 시작 yaw 오차 0.013도.
- 공통 궤적 3.501m, 거리 기준 위치 오차 중앙값 0.018m, p95 0.072m.
- actor 52는 60/60 frames에서 1--5 semantic points로 관측됨.
- ego-motion 보정 후 actor point 중앙값은 3 scan에서 2.8배, 5 scan에서
  4.17배로 증가함.

`traversability_temporal_core.py`는 과거 scan을 odometry로 현재 LiDAR 좌표계에
정렬하고 byte-identical scan을 제거한다. 표준 4-channel BEV 외에 cell별 distinct
scan support count도 반환해 단순 point-density와 시간적으로 반복된 지지를 구분할
수 있게 한다. semantic actor ID는 pilot 평가에만 사용하며 배포 입력에는 들어가지
않는다.

이 결과는 temporal 표현 실험을 진행할 근거이지 배포 승인이 아니다. 아직 회전
구간의 LiDAR extrinsic 보정, 동적 물체 ghosting, scene-wise 일반화와 기존 safety
gate를 검증하지 않았다. 다음 구현은 기존 single-scan baseline을 보존한 별도
temporal dataset/model variant이며, 비교 평가가 끝날 때까지 navigation authority와
연결하지 않는다.

Pilot report:
`/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/temporal_pilot_scene09_poseA_motorhelmet_20260921.json`

### Temporal representation archive

세 개의 scene09 motorhelmet 위치(center, left 1.2m, right 1.2m)가 모두 pilot
gate를 통과한 뒤 17-channel 실험 변형을 구현했다.

- channels 0--7: 현재 single-scan evidence. 기존 baseline과 safety 입력을 보존한다.
- channels 8--15: 고유한 최근 5 scans를 현재 LiDAR frame에 정렬한 evidence.
- channel 16: 각 cell을 지지한 distinct scan 수를 5로 나눈 support fraction.
- target: 항상 현재 semantic scan에서만 생성한다. 과거 label이나 actor ID는 모델
  입력으로 사용하지 않는다.
- exact repeated scan은 history와 출력 sample에서 제거한다.
- window age와 yaw span을 제한하며 LiDAR extrinsic을 명시적으로 보존한다.

현재 CARLA Lincoln의 `base_link -> lidar_3d_frame` TF는
`[-0.0062155, 0.0, 1.8053046, 0.0 rad]`이다. 이 값을 사용해 scene09의 control과
세 obstacle session을 비파괴 변환했다. 총 147 samples가 작성됐으며 모두
`(17, 160, 160)`, finite, 고유 history 5개이다. obstacle session의 109 samples는
모두 현재 frame에 controlled actor target이 존재한다.

Archive:
`/home/sukja/terrain_nav_data/learning/traversability/v2_temporal_scene09_poseA_20260921`

scene09 한 장면만으로 모델을 학습하거나 baseline과 일반화 성능을 비교하지 않는다.
다음 수집은 서로 다른 development 배경의 matched moving control/object 묶음이어야
하고, scene-wise split을 만들 수 있을 때에만 temporal U-Net 학습을 시작한다.

### Scene09--11 frozen temporal representation experiment

scene10은 주차 차량과 볼라드가 많은 좁은 도로, scene11은 주차 차량이
적고 인도 시설물과 열린 공간이 함께 있는 도로에서 각각 control/center/
left/right moving 세션을 수집했다. 모든 채택 obstacle 세션은 시작점,
공통 궤적, actor 관측률, 5-scan evidence gain gate를 통과했다.

- scene09 archive: 147 samples
- scene10 archive: 177 samples
- scene11 archive: 200 samples

기존 정지 paired trainer는 scene/pair group당 control이 하나임을 가정하므로,
연속 주행 frame을 하나의 control에 억지로 묶지 않는다.
`prepare_traversability_temporal_experiment` tool이 각 controlled frame을 세계
좌표에서 가장 가까운 control frame에 매칭하고 0.20m/1deg gate를
적용한다. 평가기의 prediction key도 session이 아닌 `(session, sample_id)`로
변경해 연속 frame이 서로 덮어쓰이지 않게 했다.

고정 manifest는 scene09--11의 496 rows를 포함한다.

- matched control rows: 117
- controlled rows: 379
- scene11에서 control 궤적의 끝을 0.20m 이상 넘은 6 frames만 제외
- scene09/10/11 match position p95: 0.155m / 0.086m / 0.091m
- scene08 access: false

같은 derived sample과 target, pair, fold를 공유하고 입력 view만 변경한다.

- `current_only`: `current_lidar_evidence_bev` 8 channels
- `temporal`: `lidar_evidence_bev` 17 channels

Frozen experiment:
`/home/sukja/terrain_nav_data/learning/traversability/v2_temporal_representation_experiment_scenes09_11_20260921`

CPU 1-epoch/1-train-batch smoke test는 두 view에서 모두 checkpoint 생성과
scene11 193-row 평가를 완주했다. 이 수치는 pipeline 연결 검사일 뿐
성능 결론으로 사용하지 않는다.

### Scene09--11 frozen temporal representation result

샌드박스 내부에서는 CUDA device가 가려졌지만 호스트에서 RTX 3060 12GB와
CUDA PyTorch를 확인했다. 이후 같은 496 rows, target, pair, scene fold,
architecture, threshold, seed 정책을 고정하고 입력 view만 바꿔 두 변형을
GPU로 완전 학습했다. 각 fold split manifest가 byte-identical임도 확인했다.

| 3-fold development metric | current-only 8ch | temporal 17ch | temporal delta |
| --- | ---: | ---: | ---: |
| raw obstacle IoU mean | 0.8612 | 0.8599 | -0.0013 |
| raw obstacle IoU minimum | 0.7717 | 0.7747 | +0.0030 |
| raw passable IoU mean | 0.9600 | 0.9585 | -0.0015 |
| raw passable IoU minimum | 0.9494 | 0.9377 | -0.0116 |
| controlled-cell recall mean | 0.9968 | 0.9859 | -0.0109 |
| controlled-instance recall mean | 0.9946 | 0.9943 | -0.0003 |
| paired-control obstacle probability mean | 0.0046 | 0.0059 | +0.0013 |
| paired obstacle-control delta mean | 0.9908 | 0.9620 | -0.0288 |
| fused obstacle IoU mean | 0.8320 | 0.8390 | +0.0071 |
| fused controlled-instance recall mean | 0.9969 | 1.0000 | +0.0031 |

두 변형 모두 controlled unsafe-passable instance 0과 paired-control fused false
obstacle 0을 유지했다. temporal raw obstacle IoU는 scene09에서 +0.0266,
scene11에서 +0.0030이었지만 scene10에서 -0.0335였다. passable IoU도
scene10에서 -0.0271로 하락했다. 즉 장면별 이득이 일관되지 않고, 평균 raw
obstacle/passable IoU와 controlled-cell recall 및 paired separation이 개선되지
않았다.

따라서 현재의 early-concatenated accumulated-BEV 17-channel 입력은
current-only를 대체하지 않는다. 이 결과는 temporal evidence 일반을 기각하는
것이 아니라, 현재 누적 표현과 단순 입력 결합이 일반화 이득을 만들지 못했다는
결론이다. current-only를 development representation으로 유지하되 둘 다
`deployable=false`, navigation authority 미연결 상태를 유지한다. 다음 실험은
현재 frame encoder와 temporal residual/persistence encoder를 분리해 scene10
회귀 없이 장면 간 일반화를 개선하는지를 같은 frozen protocol로 확인한다.

결과 경로:

- current-only CV:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/cv_temporal_representation_current_only_scenes09_11_20260921`
- temporal CV:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/cv_temporal_representation_17ch_scenes09_11_20260921`
- selection record:
  `/home/sukja/terrain_nav_data/learning/models/traversability_evidence_v2/temporal_representation_selection_scenes09_11_20260921.json`
- MLflow reference-only comparison run:
  `4d0649ec554641369f811139847245fc`

## 2026-09-22 current-only 8-channel online shadow inference

선택한 current-only 표현을 실제 `/lidar/points`에서 실행하는
`traversability_evidence_shadow_node`를 추가했다. 이 노드는 다음 topic만
발행하며 Nav2, path-clearance, emergency stop, `/cmd_vel`에는 연결하지 않는다.

- `/learning/evidence_v2/passable_probability`
- `/learning/evidence_v2/obstacle_probability`
- `/learning/evidence_v2/decision`
- `/learning/evidence_v2/hard_obstacle`
- `/learning/evidence_v2/status`

학습 archive는 4-channel BEV를 float16으로 저장한 뒤 그 저장본에서 8-channel
evidence를 만들고, evidence도 float16으로 저장한다. 온라인 전처리는 두 정밀도
경계를 모두 재현한 후 float32로 추론한다. scene09--11에서 각각 앞/중간/끝
sample을 고른 9개 대조에서 archived BEV, evidence, passable probability,
obstacle probability의 최대 오차는 모두 정확히 0.0이었다.

현재 기본 checkpoint는 online wiring과 shadow 주행을 위한 scene11 holdout fold
checkpoint다. 세 development scene 전체를 사용한 최종 active checkpoint는
아니므로, 이후 add-only authority를 켜기 전에 별도로 고정해야 한다.

CPU에서 실제 10,092-point scan을 20회 측정한 중앙값은 전처리 42.78 ms,
model 11.09 ms, 합계 53.93 ms였다. 기본 5 Hz shadow 실행에는 충분하다.

```bash
cd /home/sukja/terrain_nav_ws
source install/setup.bash

ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=false \
  record_learning_data:=false \
  traversability_shadow_enabled:=false \
  traversability_evidence_v2_enabled:=true \
  traversability_evidence_v2_device:=auto \
  traversability_evidence_v2_mc_samples:=1
```

`mc_samples:=1`은 frozen offline 평가와 동일한 deterministic 출력 계약이다.
float probability/variance image가 필요할 때만
`traversability_evidence_v2_diagnostics_enabled:=true`를 사용한다. 상태 JSON의
`navigation_control_effect`는 현재 항상 `none`이다.

offline/online parity를 다시 검사하는 예시는 다음과 같다.

```bash
ros2 run terrain_navigation_pkg \
  validate_traversability_evidence_online_parity \
  --source-sample /absolute/raw/session/samples/sample.npz \
  --derived-sample /absolute/derived/session/sample.npz \
  --checkpoint /absolute/current_only/best.pt
```

다음 단계는 이 노드의 출력을 기존 baseline cloud와 결합하는 별도 fail-safe
fusion이다. 첫 authority는 obstacle을 추가만 할 수 있고 passable/unknown은 기존
baseline obstacle을 지우지 못한다. AI 출력이 stale이거나 노드가 죽으면 매 scan
baseline을 그대로 내보내야 한다. 원본 LiDAR emergency stop은 계속 독립시킨다.

### Fail-safe obstacle authority

`traversability_obstacle_authority_node`가 기존 filter의 private baseline
`/lidar/nav2_obstacles_baseline`과 v2 decision을 받아 Nav2와 path-clearance가 함께
사용하는 `/lidar/nav2_obstacles`를 발행한다. 지원 mode는 다음 셋이다.

- `baseline`: AI를 보지 않고 baseline을 그대로 relay한다.
- `passive`: selected output은 baseline 그대로이며, exact-stamp AI candidate만
  `/learning/evidence_v2/nav2_obstacles_candidate`에 별도로 발행한다.
- `add_only`: usable AI가 있을 때 baseline 전체와 새로운 obstacle endpoint를
  합친 selected cloud를 발행한다. AI 시작 전, 0.5초 이상 침묵, stamp match 지연,
  schema/frame 불일치 또는 fusion exception이면 baseline으로 복귀한다.

최종 physical gate와 voxel 중복 제거까지 통과해 selected cloud에 실제로 append된
endpoint만 `/learning/evidence_v2/added_obstacles`에도 발행한다. 이 진단 cloud의
point record는 selected cloud 뒤쪽에 붙은 record와 동일하다. baseline fallback과
passive mode에서는 빈 cloud를 발행하므로 RViz가 이전 AI 추가점을 남겨 보이지
않는다.

add-only 출력은 baseline point record 전체를 순서와 값 그대로 앞에 유지한다.
voxel 중복 제거는 추가 후보에만 적용하므로 baseline 삭제 개수는 구조상 항상
0이다. passable과 unknown은 baseline을 변경하지 않는다. obstacle cell에서는
cell maximum height의 5cm 이내 endpoint 중, 0.75m local ground가 유효하고 그
지면보다 최소 7cm 높은 점만 추가한다. 따라서 flat-road false positive의 유일한
return을 obstacle로 승격하지 않는다. AI의 추가 권한은 초기 검증 범위 12m로
제한하지만 기존 baseline은 계속 45m까지 감지한다. 선택 node는 model과 분리되어
있어 model process가 죽어도 baseline relay를 계속하며 launch는 선택 node 자체도
1초 간격으로 respawn한다.

`selective_clear`는 의도적으로 노출하지 않았다. add-only A/B 검증이 끝나기 전에는
학습 결과가 기존 장애물을 지울 이유가 없다. emergency stop은 계속 원본
`/lidar/points`를 직접 사용하므로 이 authority mode의 영향을 받지 않는다.
