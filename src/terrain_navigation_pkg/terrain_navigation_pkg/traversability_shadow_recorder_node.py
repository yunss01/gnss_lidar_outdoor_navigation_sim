#!/usr/bin/env python3
"""Record learned traversability diagnostics without control authority."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import Bool, Float32, String, UInt32

from .traversability_shadow_recorder_core import (
    enrich_shadow_status,
    summarize_shadow_rows,
)


CSV_FIELDS = [
    'sample_id', 'wall_time_iso', 'ros_time_s', 'route_elapsed_s',
    'source_stamp_s', 'source_frame', 'route_status', 'route_index',
    'route_size', 'distance_to_goal_m', 'speed_mps', 'safety_state',
    'emergency_stop', 'nav2_status', 'far_guide_status', 'checkpoint',
    'device', 'mc_samples', 'scan_points', 'observed_cells',
    'learned_free_cells', 'learned_obstacle_cells',
    'hard_obstacle_cells', 'uncertain_cells', 'known_cells',
    'obstacle_cells', 'known_coverage', 'free_fraction',
    'obstacle_fraction', 'uncertain_fraction', 'mean_entropy',
    'mean_mc_variance', 'inference_ms', 'snapshot_file',
]


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    temporary.replace(path)


def _stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


class TraversabilityShadowRecorder(Node):
    """Write compact route diagnostics and sparse grid snapshots."""

    def __init__(self):
        super().__init__('traversability_shadow_recorder_node')
        self._declare_parameters()
        self.output_root = Path(str(
            self.get_parameter('output_directory').value
        )).expanduser()
        self.snapshot_interval_s = float(
            self.get_parameter('snapshot_interval_s').value
        )
        self.high_uncertainty_fraction = float(
            self.get_parameter('high_uncertainty_fraction').value
        )
        self.high_uncertainty_cooldown_s = float(
            self.get_parameter('high_uncertainty_cooldown_s').value
        )
        self.maximum_snapshots = int(
            self.get_parameter('maximum_snapshots_per_route').value
        )
        self.maximum_grid_alignment_s = float(
            self.get_parameter('maximum_grid_alignment_s').value
        )
        if self.snapshot_interval_s <= 0.0 or self.maximum_snapshots < 1:
            raise ValueError('snapshot interval and maximum must be positive')

        self.route_status = 'idle'
        self.route_index = 0
        self.route_size = 0
        self.latest = {
            'distance_to_goal_m': math.nan,
            'speed_mps': math.nan,
            'safety_state': '',
            'emergency_stop': False,
            'nav2_status': '',
            'far_guide_status': '',
        }
        self.grids = {}
        self.active_path = None
        self.csv_file = None
        self.csv_writer = None
        self.started_monotonic = None
        self.started_at = None
        self.rows = []
        self.sample_id = 0
        self.snapshot_count = 0
        self.last_snapshot_monotonic = -math.inf
        self.last_uncertainty_snapshot_monotonic = -math.inf
        self.last_snapshot_route_index = None
        self.last_frame_sequence = None

        self._create_subscriptions()
        self.get_logger().info(
            'Traversability shadow recorder ready: output=%s, snapshots '
            'every %.1f s (max %d); NO control authority' % (
                self.output_root, self.snapshot_interval_s,
                self.maximum_snapshots,
            )
        )

    def _declare_parameters(self):
        defaults = {
            'output_directory': '~/terrain_nav_data/learning/shadow_runs',
            'snapshot_interval_s': 10.0,
            'high_uncertainty_fraction': 0.15,
            'high_uncertainty_cooldown_s': 5.0,
            'maximum_snapshots_per_route': 64,
            'maximum_grid_alignment_s': 0.20,
            'shadow_status_topic': (
                '/learning/traversability_shadow_status'
            ),
            'probability_topic': (
                '/learning/traversability_probability'
            ),
            'uncertainty_topic': (
                '/learning/traversability_uncertainty'
            ),
            'decision_topic': (
                '/learning/traversability_shadow_decision'
            ),
            'route_status_topic': '/navigation/route_status',
            'route_index_topic': '/navigation/route_index',
            'route_size_topic': '/navigation/route_size',
            'distance_to_goal_topic': '/navigation/distance_to_goal',
            'odometry_topic': '/vehicle/odometry',
            'safety_state_topic': '/safety/state',
            'emergency_stop_topic': '/safety/emergency_stop',
            'nav2_status_topic': '/navigation/nav2_status',
            'far_guide_status_topic': '/navigation/far_guide_status',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _create_subscriptions(self):
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        def topic(name):
            return str(self.get_parameter(name).value)

        self.create_subscription(
            String, topic('shadow_status_topic'), self._on_shadow_status,
            reliable,
        )
        for key, parameter in (
            ('probability', 'probability_topic'),
            ('uncertainty', 'uncertainty_topic'),
            ('decision', 'decision_topic'),
        ):
            self.create_subscription(
                OccupancyGrid, topic(parameter),
                lambda message, name=key: self._on_grid(name, message),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            String, topic('route_status_topic'), self._on_route_status,
            latched,
        )
        self.create_subscription(
            UInt32, topic('route_index_topic'),
            lambda message: setattr(self, 'route_index', int(message.data)),
            latched,
        )
        self.create_subscription(
            UInt32, topic('route_size_topic'),
            lambda message: setattr(self, 'route_size', int(message.data)),
            latched,
        )
        self.create_subscription(
            Odometry, topic('odometry_topic'), self._on_odometry, reliable
        )
        for message_type, parameter, key, transform in (
            (Float32, 'distance_to_goal_topic', 'distance_to_goal_m', float),
            (String, 'safety_state_topic', 'safety_state', str),
            (Bool, 'emergency_stop_topic', 'emergency_stop', bool),
            (String, 'nav2_status_topic', 'nav2_status', str),
            (String, 'far_guide_status_topic', 'far_guide_status', str),
        ):
            self.create_subscription(
                message_type, topic(parameter),
                lambda message, name=key, fn=transform: (
                    self.latest.__setitem__(name, fn(message.data))
                ),
                reliable,
            )

    def _on_odometry(self, message):
        velocity = message.twist.twist.linear
        self.latest['speed_mps'] = math.sqrt(
            velocity.x * velocity.x
            + velocity.y * velocity.y
            + velocity.z * velocity.z
        )

    def _on_grid(self, name, message):
        self.grids[name] = message

    def _on_route_status(self, message):
        status = str(message.data).strip().lower()
        previous = self.route_status
        self.route_status = status
        if status == 'navigating' and self.active_path is None:
            self._start_session()
        elif self.active_path is not None and status in {
            'completed', 'final_nav2_failed', 'invalid_route',
        }:
            self._close_session(status)
        elif (
            self.active_path is not None
            and status in {'idle', 'ready'}
            and previous not in {'idle', 'ready', 'waiting_for_goal_manager'}
        ):
            self._close_session('interrupted_' + status)

    def _start_session(self):
        now = datetime.now(timezone.utc).astimezone()
        name = now.strftime('shadow_%Y%m%d_%H%M%S_%f')
        path = self.output_root / name
        path.mkdir(parents=True, exist_ok=False)
        (path / 'snapshots').mkdir()
        csv_file = (path / 'frames.csv').open(
            'w', newline='', encoding='utf-8'
        )
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()
        self.active_path = path
        self.csv_file = csv_file
        self.csv_writer = writer
        self.started_monotonic = time.monotonic()
        self.started_at = now.isoformat()
        self.rows = []
        self.sample_id = 0
        self.snapshot_count = 0
        self.last_snapshot_monotonic = -math.inf
        self.last_uncertainty_snapshot_monotonic = -math.inf
        self.last_snapshot_route_index = None
        self.last_frame_sequence = None
        _write_json(path / 'metadata.json', {
            'schema_version': 1,
            'session': name,
            'started_at': self.started_at,
            'purpose': 'route-level learned traversability shadow audit',
            'control_effect': 'none; subscriber-only recorder',
            'speed_source': (
                '3D linear-velocity magnitude from /vehicle/odometry (m/s)'
            ),
            'route_size_at_start': self.route_size,
            'snapshot_policy': {
                'periodic_interval_s': self.snapshot_interval_s,
                'high_uncertainty_fraction': self.high_uncertainty_fraction,
                'high_uncertainty_cooldown_s': (
                    self.high_uncertainty_cooldown_s
                ),
                'maximum_snapshots_per_route': self.maximum_snapshots,
            },
        })
        self.get_logger().info('Shadow recording started: ' + str(path))

    def _on_shadow_status(self, message):
        if self.active_path is None or self.csv_writer is None:
            return
        try:
            raw_status = json.loads(message.data)
        except (TypeError, json.JSONDecodeError) as error:
            self.get_logger().warning(
                'Invalid shadow status JSON: ' + str(error)
            )
            return
        sequence = raw_status.get('frame_sequence')
        if sequence is not None and sequence == self.last_frame_sequence:
            return
        self.last_frame_sequence = sequence
        status = enrich_shadow_status(raw_status)
        self.sample_id += 1
        now_monotonic = time.monotonic()
        row = {name: '' for name in CSV_FIELDS}
        row.update(status)
        row.update(self.latest)
        row.update({
            'sample_id': self.sample_id,
            'wall_time_iso': datetime.now(
                timezone.utc
            ).astimezone().isoformat(),
            'ros_time_s': self.get_clock().now().nanoseconds * 1.0e-9,
            'route_elapsed_s': now_monotonic - self.started_monotonic,
            'route_status': self.route_status,
            'route_index': self.route_index,
            'route_size': self.route_size,
            'snapshot_file': '',
        })
        reason = self._snapshot_reason(row, now_monotonic)
        if reason:
            relative = self._save_snapshot(row, reason)
            if relative:
                row['snapshot_file'] = relative
                self.last_snapshot_monotonic = now_monotonic
                self.last_snapshot_route_index = self.route_index
                if reason == 'high_uncertainty':
                    self.last_uncertainty_snapshot_monotonic = now_monotonic
        filtered = {name: row.get(name, '') for name in CSV_FIELDS}
        self.csv_writer.writerow(filtered)
        self.csv_file.flush()
        self.rows.append(filtered)

    def _snapshot_reason(self, row, now):
        if self.snapshot_count >= self.maximum_snapshots:
            return ''
        if self.sample_id == 1:
            return 'route_start'
        if self.route_index != self.last_snapshot_route_index:
            return 'waypoint_change'
        uncertain = float(row.get('uncertain_fraction', 0.0))
        if (
            uncertain >= self.high_uncertainty_fraction
            and now - self.last_uncertainty_snapshot_monotonic
            >= self.high_uncertainty_cooldown_s
        ):
            return 'high_uncertainty'
        if now - self.last_snapshot_monotonic >= self.snapshot_interval_s:
            return 'periodic'
        return ''

    def _save_snapshot(self, row, reason):
        required = ('probability', 'uncertainty', 'decision')
        if any(name not in self.grids for name in required):
            return ''
        source_stamp = float(row.get('source_stamp_s') or math.nan)
        if math.isfinite(source_stamp):
            deltas = [
                abs(
                    _stamp_seconds(self.grids[name].header.stamp)
                    - source_stamp
                )
                for name in required
            ]
            if max(deltas) > self.maximum_grid_alignment_s:
                return ''
        arrays = {}
        for name in required:
            message = self.grids[name]
            expected = int(message.info.width) * int(message.info.height)
            values = np.asarray(message.data, dtype=np.int8)
            if expected == 0 or values.size != expected:
                return ''
            arrays[name] = values.reshape(
                (int(message.info.height), int(message.info.width))
            )
        self.snapshot_count += 1
        filename = 'sample_{:06d}_wp{:02d}_{}.npz'.format(
            self.sample_id, self.route_index, reason
        )
        relative = Path('snapshots') / filename
        destination = self.active_path / relative
        temporary = destination.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(
                stream,
                probability=arrays['probability'],
                uncertainty=arrays['uncertainty'],
                decision=arrays['decision'],
                status_json=np.asarray(json.dumps(row, sort_keys=True)),
            )
        temporary.replace(destination)
        return str(relative)

    def _close_session(self, result):
        if self.active_path is None:
            return
        path = self.active_path
        ended_at = datetime.now(timezone.utc).astimezone().isoformat()
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
        summary = summarize_shadow_rows(self.rows)
        summary.update({
            'result': result,
            'started_at': self.started_at,
            'ended_at': ended_at,
            'route_status': self.route_status,
            'route_index': self.route_index,
            'route_size': self.route_size,
            'snapshot_count': self.snapshot_count,
            'note': (
                'Shadow metrics measure confidence and decision coverage, '
                'not ground-truth perception accuracy.'
            ),
        })
        _write_json(path / 'summary.json', summary)
        self.get_logger().info(
            'Shadow recording closed: result=%s frames=%d snapshots=%d path=%s'
            % (result, len(self.rows), self.snapshot_count, path)
        )
        self.active_path = None
        self.csv_file = None
        self.csv_writer = None
        self.rows = []

    def close(self):
        if self.active_path is not None:
            self._close_session('node_shutdown')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TraversabilityShadowRecorder()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
