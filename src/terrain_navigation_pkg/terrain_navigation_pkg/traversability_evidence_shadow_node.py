#!/usr/bin/env python3
"""Run current-only v2 evidence online with no navigation authority."""

from __future__ import annotations

import hashlib
import json
import time

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String
import torch

from .navigation_learning_recorder_core import BevGeometry
from .traversability_evidence_core import (
    build_conservative_evidence_decision,
)
from .traversability_evidence_inference_core import (
    build_current_evidence_input,
    load_current_only_evidence_checkpoint,
    predict_evidence,
)
from .traversability_inference_core import (
    bev_to_occupancy_grid,
    inference_rate_limit_allows,
)
from .traversability_learning_core import EgoFootprint


DEFAULT_CHECKPOINT = (
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/'
    'cv_temporal_representation_current_only_scenes09_11_20260921/'
    'holdout_scene11/model/best.pt'
)


def _stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class TraversabilityEvidenceShadowNode(Node):
    """Publish v2 passable/obstacle/unknown grids without control authority."""

    def __init__(self):
        super().__init__('traversability_evidence_shadow_node')
        self.declare_parameter('checkpoint_path', DEFAULT_CHECKPOINT)
        self.declare_parameter('input_topic', '/lidar/points')
        self.declare_parameter(
            'passable_probability_topic',
            '/learning/evidence_v2/passable_probability',
        )
        self.declare_parameter(
            'obstacle_probability_topic',
            '/learning/evidence_v2/obstacle_probability',
        )
        self.declare_parameter(
            'decision_topic', '/learning/evidence_v2/decision'
        )
        self.declare_parameter(
            'hard_obstacle_topic', '/learning/evidence_v2/hard_obstacle'
        )
        self.declare_parameter(
            'status_topic', '/learning/evidence_v2/status'
        )
        self.declare_parameter('device', 'auto')
        self.declare_parameter('mc_samples', 1)
        self.declare_parameter('maximum_inference_rate_hz', 5.0)
        self.declare_parameter('publish_diagnostic_images', False)
        self.declare_parameter(
            'passable_probability_float_topic',
            '/learning/evidence_v2/passable_probability_float',
        )
        self.declare_parameter(
            'obstacle_probability_float_topic',
            '/learning/evidence_v2/obstacle_probability_float',
        )
        self.declare_parameter(
            'passable_variance_float_topic',
            '/learning/evidence_v2/passable_variance_float',
        )
        self.declare_parameter(
            'obstacle_variance_float_topic',
            '/learning/evidence_v2/obstacle_variance_float',
        )

        self.declare_parameter('passable_probability_threshold', 0.90)
        self.declare_parameter('obstacle_probability_threshold', 0.50)
        self.declare_parameter(
            'maximum_obstacle_probability_for_passable', 0.10
        )
        self.declare_parameter('minimum_passable_support', 0.20)
        self.declare_parameter('maximum_mc_variance', 0.02)
        self.declare_parameter(
            'hard_obstacle_minimum_relative_height_m', 0.15
        )
        self.declare_parameter(
            'hard_obstacle_minimum_vertical_span_m', 0.15
        )
        self.declare_parameter(
            'hard_obstacle_minimum_absolute_height_m', -1.40
        )
        self.declare_parameter('local_ground_near_radius_m', 0.75)
        self.declare_parameter('local_ground_far_radius_m', 1.50)
        self.declare_parameter('local_ground_quantile', 0.25)
        self.declare_parameter('local_ground_minimum_support_cells', 4)

        self.declare_parameter('bev_x_min_m', -10.0)
        self.declare_parameter('bev_x_max_m', 30.0)
        self.declare_parameter('bev_y_min_m', -20.0)
        self.declare_parameter('bev_y_max_m', 20.0)
        self.declare_parameter('bev_resolution_m', 0.25)
        self.declare_parameter('bev_z_min_m', -2.0)
        self.declare_parameter('bev_z_max_m', 3.0)
        self.declare_parameter('ego_rear_m', 2.5)
        self.declare_parameter('ego_front_m', 2.4)
        self.declare_parameter('ego_half_width_m', 1.0)

        self.geometry = BevGeometry(
            x_min_m=float(self.get_parameter('bev_x_min_m').value),
            x_max_m=float(self.get_parameter('bev_x_max_m').value),
            y_min_m=float(self.get_parameter('bev_y_min_m').value),
            y_max_m=float(self.get_parameter('bev_y_max_m').value),
            resolution_m=float(
                self.get_parameter('bev_resolution_m').value
            ),
            z_min_m=float(self.get_parameter('bev_z_min_m').value),
            z_max_m=float(self.get_parameter('bev_z_max_m').value),
        )
        self.footprint = EgoFootprint(
            rear_m=float(self.get_parameter('ego_rear_m').value),
            front_m=float(self.get_parameter('ego_front_m').value),
            half_width_m=float(
                self.get_parameter('ego_half_width_m').value
            ),
        )
        self.mc_samples = int(self.get_parameter('mc_samples').value)
        rate = float(
            self.get_parameter('maximum_inference_rate_hz').value
        )
        if self.mc_samples < 1 or rate <= 0.0:
            raise ValueError('mc_samples and inference rate must be positive')
        self.minimum_interval_s = 1.0 / rate
        self.last_inference_source_s = -np.inf
        self.frame_sequence = 0
        self.publish_diagnostics = bool(
            self.get_parameter('publish_diagnostic_images').value
        )

        requested_device = str(self.get_parameter('device').value)
        if requested_device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('device must be auto, cpu, or cuda')
        device_name = requested_device
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_name == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable to PyTorch')
        self.device = torch.device(device_name)
        self.model, checkpoint, self.checkpoint_path = (
            load_current_only_evidence_checkpoint(
                str(self.get_parameter('checkpoint_path').value),
                self.device,
            )
        )
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        self.model_config = dict(checkpoint['model_config'])

        self.passable_publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('passable_probability_topic').value),
            qos_profile_sensor_data,
        )
        self.obstacle_publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('obstacle_probability_topic').value),
            qos_profile_sensor_data,
        )
        self.decision_publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('decision_topic').value),
            qos_profile_sensor_data,
        )
        self.hard_obstacle_publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('hard_obstacle_topic').value),
            qos_profile_sensor_data,
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter('status_topic').value),
            10,
        )
        self.diagnostic_publishers = {}
        if self.publish_diagnostics:
            for name, parameter in (
                ('passable', 'passable_probability_float_topic'),
                ('obstacle', 'obstacle_probability_float_topic'),
                ('passable_variance', 'passable_variance_float_topic'),
                ('obstacle_variance', 'obstacle_variance_float_topic'),
            ):
                self.diagnostic_publishers[name] = self.create_publisher(
                    Image,
                    str(self.get_parameter(parameter).value),
                    qos_profile_sensor_data,
                )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('input_topic').value),
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            'Traversability evidence v2 shadow ready: device={}, MC={}, '
            'rate={:.1f} Hz, architecture={}; NO navigation authority'.format(
                self.device,
                self.mc_samples,
                rate,
                self.model_config.get('architecture'),
            )
        )

    @staticmethod
    def _float32_view(message, name):
        field = next(
            (item for item in message.fields if item.name == name), None
        )
        if field is None or field.datatype != PointField.FLOAT32:
            raise ValueError('{} FLOAT32 field is required'.format(name))
        endian = '>f4' if message.is_bigendian else '<f4'
        count = int(message.width) * int(message.height)
        return np.ndarray(
            shape=(count,),
            dtype=np.dtype(endian),
            buffer=message.data,
            offset=int(field.offset),
            strides=(int(message.point_step),),
        )

    @staticmethod
    def _image_message(values, source):
        array = np.ascontiguousarray(values, dtype='<f4')
        output = Image()
        output.header = source.header
        output.height = int(array.shape[0])
        output.width = int(array.shape[1])
        output.encoding = '32FC1'
        output.is_bigendian = False
        output.step = int(output.width) * 4
        output.data = array.tobytes()
        return output

    def _grid_message(self, values, source):
        converted = bev_to_occupancy_grid(values).astype(
            np.int8, copy=False
        )
        output = OccupancyGrid()
        output.header = source.header
        output.info.resolution = float(self.geometry.resolution_m)
        output.info.width = int(self.geometry.height)
        output.info.height = int(self.geometry.width)
        output.info.origin.position.x = float(self.geometry.x_min_m)
        output.info.origin.position.y = float(self.geometry.y_min_m)
        output.info.origin.orientation.w = 1.0
        output.data = converted.ravel().tolist()
        return output

    @staticmethod
    def _probability_grid(probability, available):
        grid = np.full(probability.shape, -1, dtype=np.int8)
        grid[available] = np.rint(
            100.0 * np.asarray(probability)[available]
        ).astype(np.int8)
        return grid

    def _on_cloud(self, message):
        source_stamp_s = _stamp_seconds(message.header.stamp)
        if not inference_rate_limit_allows(
            source_stamp_s,
            self.last_inference_source_s,
            self.minimum_interval_s,
        ):
            return
        self.last_inference_source_s = source_stamp_s
        started = time.perf_counter()
        try:
            points = np.column_stack((
                self._float32_view(message, 'x'),
                self._float32_view(message, 'y'),
                self._float32_view(message, 'z'),
            ))
            prepared = build_current_evidence_input(
                points,
                self.geometry,
                self.footprint,
                local_ground_radii_m=(
                    float(self.get_parameter(
                        'local_ground_near_radius_m'
                    ).value),
                    float(self.get_parameter(
                        'local_ground_far_radius_m'
                    ).value),
                ),
                local_ground_quantile=float(self.get_parameter(
                    'local_ground_quantile'
                ).value),
                local_ground_minimum_support_cells=int(self.get_parameter(
                    'local_ground_minimum_support_cells'
                ).value),
                hard_obstacle_minimum_relative_height_m=float(
                    self.get_parameter(
                        'hard_obstacle_minimum_relative_height_m'
                    ).value
                ),
                hard_obstacle_minimum_vertical_span_m=float(
                    self.get_parameter(
                        'hard_obstacle_minimum_vertical_span_m'
                    ).value
                ),
                hard_obstacle_minimum_absolute_height_m=float(
                    self.get_parameter(
                        'hard_obstacle_minimum_absolute_height_m'
                    ).value
                ),
            )
            preprocessed_at = time.perf_counter()
            prediction = predict_evidence(
                self.model,
                prepared.normalized_evidence_bev,
                self.device,
                mc_samples=self.mc_samples,
            )
            predicted_at = time.perf_counter()
            use_uncertainty = self.mc_samples > 1
            decision = build_conservative_evidence_decision(
                prediction.passable_probability,
                prediction.obstacle_probability,
                prepared.observed_mask,
                prepared.passable_support,
                passable_uncertainty=(
                    prediction.passable_variance
                    if use_uncertainty else None
                ),
                obstacle_uncertainty=(
                    prediction.obstacle_variance
                    if use_uncertainty else None
                ),
                hard_obstacle_mask=prepared.hard_obstacle_mask,
                obstacle_probability_threshold=float(self.get_parameter(
                    'obstacle_probability_threshold'
                ).value),
                passable_probability_threshold=float(self.get_parameter(
                    'passable_probability_threshold'
                ).value),
                maximum_obstacle_probability_for_passable=float(
                    self.get_parameter(
                        'maximum_obstacle_probability_for_passable'
                    ).value
                ),
                minimum_passable_support=float(self.get_parameter(
                    'minimum_passable_support'
                ).value),
                maximum_uncertainty=float(self.get_parameter(
                    'maximum_mc_variance'
                ).value),
            )
        except (KeyError, OSError, ValueError, RuntimeError) as error:
            self.get_logger().error(
                'evidence v2 shadow inference failed: ' + str(error)
            )
            return

        available = prepared.observed_mask
        self.passable_publisher.publish(self._grid_message(
            self._probability_grid(
                prediction.passable_probability, available
            ),
            message,
        ))
        self.obstacle_publisher.publish(self._grid_message(
            self._probability_grid(
                prediction.obstacle_probability, available
            ),
            message,
        ))
        self.decision_publisher.publish(
            self._grid_message(decision.decision, message)
        )
        hard_grid = np.full(available.shape, -1, dtype=np.int8)
        hard_grid[available] = 0
        hard_grid[decision.hard_obstacle_mask] = 100
        self.hard_obstacle_publisher.publish(
            self._grid_message(hard_grid, message)
        )
        if self.publish_diagnostics:
            values = {
                'passable': prediction.passable_probability,
                'obstacle': prediction.obstacle_probability,
                'passable_variance': prediction.passable_variance,
                'obstacle_variance': prediction.obstacle_variance,
            }
            for name, array in values.items():
                self.diagnostic_publishers[name].publish(
                    self._image_message(array, message)
                )

        self.frame_sequence += 1
        completed_at = time.perf_counter()
        status = {
            'mode': 'evidence_v2_shadow_no_control_authority',
            'navigation_control_effect': 'none',
            'checkpoint': str(self.checkpoint_path),
            'checkpoint_sha256': self.checkpoint_sha256,
            'architecture': self.model_config.get('architecture'),
            'input_variant': 'current_only',
            'input_channels': 8,
            'device': str(self.device),
            'mc_samples': self.mc_samples,
            'frame_sequence': self.frame_sequence,
            'source_stamp_s': source_stamp_s,
            'source_frame': str(message.header.frame_id),
            'scan_points': int(points.shape[0]),
            'observed_cells': int(np.count_nonzero(available)),
            'hard_obstacle_cells': int(np.count_nonzero(
                decision.hard_obstacle_mask
            )),
            'learned_obstacle_cells': int(np.count_nonzero(
                decision.learned_obstacle_mask
            )),
            'passable_cells': int(np.count_nonzero(
                decision.passable_accepted
            )),
            'unknown_observed_cells': int(np.count_nonzero(
                decision.unknown_mask & available
            )),
            'unobserved_cells': int(np.count_nonzero(~available)),
            'preprocess_ms': 1000.0 * (preprocessed_at - started),
            'model_ms': 1000.0 * (predicted_at - preprocessed_at),
            'total_ms': 1000.0 * (completed_at - started),
        }
        output = String()
        output.data = json.dumps(status, sort_keys=True)
        self.status_publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TraversabilityEvidenceShadowNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
