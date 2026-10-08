#!/usr/bin/env python3
"""Run learned traversability online without navigation control authority."""

from __future__ import annotations

import json
from pathlib import Path
import time

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String
import torch

from .navigation_learning_recorder_core import BevGeometry, build_lidar_bev
from .traversability_dataset import normalize_lidar_bev
from .traversability_inference_core import (
    bev_to_occupancy_grid,
    binary_predictive_entropy,
    build_selective_decision,
    inference_rate_limit_allows,
)
from .traversability_learning_core import EgoFootprint, ego_footprint_mask
from .traversability_model import BevTraversabilityUNet


DEFAULT_CHECKPOINT = (
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_pilot/v2_domain_aug/best.pt'
)


def _stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


class TraversabilityShadowNode(Node):
    """Publish learned grids on isolated topics; never command the vehicle."""

    def __init__(self):
        super().__init__('traversability_shadow_node')
        self.declare_parameter('checkpoint_path', DEFAULT_CHECKPOINT)
        self.declare_parameter('input_topic', '/lidar/points')
        self.declare_parameter(
            'probability_topic', '/learning/traversability_probability'
        )
        self.declare_parameter(
            'uncertainty_topic', '/learning/traversability_uncertainty'
        )
        self.declare_parameter(
            'decision_topic', '/learning/traversability_shadow_decision'
        )
        self.declare_parameter('publish_diagnostic_images', False)
        self.declare_parameter(
            'probability_float_topic',
            '/learning/traversability_probability_float',
        )
        self.declare_parameter(
            'entropy_float_topic',
            '/learning/traversability_entropy_float',
        )
        self.declare_parameter(
            'mc_variance_float_topic',
            '/learning/traversability_mc_variance_float',
        )
        self.declare_parameter(
            'hard_obstacle_mask_topic',
            '/learning/traversability_hard_obstacle_mask',
        )
        self.declare_parameter(
            'status_topic', '/learning/traversability_shadow_status'
        )
        self.declare_parameter('device', 'auto')
        self.declare_parameter('mc_samples', 4)
        self.declare_parameter('maximum_inference_rate_hz', 5.0)
        self.declare_parameter('free_probability_threshold', 0.99)
        self.declare_parameter('obstacle_probability_threshold', 0.50)
        self.declare_parameter('maximum_entropy', 0.15)
        self.declare_parameter('maximum_mc_variance', 0.02)
        self.declare_parameter('hard_obstacle_minimum_z_m', -1.40)
        self.declare_parameter('hard_obstacle_vertical_span_m', 0.15)
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
        footprint = EgoFootprint(
            rear_m=float(self.get_parameter('ego_rear_m').value),
            front_m=float(self.get_parameter('ego_front_m').value),
            half_width_m=float(
                self.get_parameter('ego_half_width_m').value
            ),
        )
        self.ego_mask = ego_footprint_mask(self.geometry, footprint)
        self.mc_samples = int(self.get_parameter('mc_samples').value)
        rate = float(self.get_parameter('maximum_inference_rate_hz').value)
        if self.mc_samples < 1 or rate <= 0.0:
            raise ValueError('mc_samples and inference rate must be positive')
        self.minimum_interval_s = 1.0 / rate
        self.last_inference_source_s = -np.inf
        self.frame_sequence = 0
        self.publish_diagnostic_images = bool(
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

        checkpoint_path = Path(
            str(self.get_parameter('checkpoint_path').value)
        ).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                'checkpoint not found: ' + str(checkpoint_path)
            )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model = BevTraversabilityUNet(
            **checkpoint['model_config']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.checkpoint_path = checkpoint_path.resolve()

        self.probability_publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('probability_topic').value),
            qos_profile_sensor_data,
        )
        self.uncertainty_publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('uncertainty_topic').value),
            qos_profile_sensor_data,
        )
        self.decision_publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('decision_topic').value),
            qos_profile_sensor_data,
        )
        self.diagnostic_publishers = {}
        if self.publish_diagnostic_images:
            for name, parameter in (
                ('probability', 'probability_float_topic'),
                ('entropy', 'entropy_float_topic'),
                ('variance', 'mc_variance_float_topic'),
                ('hard_mask', 'hard_obstacle_mask_topic'),
            ):
                self.diagnostic_publishers[name] = self.create_publisher(
                    Image,
                    str(self.get_parameter(parameter).value),
                    qos_profile_sensor_data,
                )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter('status_topic').value), 10
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('input_topic').value),
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            'Traversability shadow ready: device={}, MC={}, rate={:.1f} Hz; '
            'diagnostics={}; NO navigation or safety authority'.format(
                self.device,
                self.mc_samples,
                rate,
                self.publish_diagnostic_images,
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
            shape=(count,), dtype=np.dtype(endian), buffer=message.data,
            offset=int(field.offset), strides=(int(message.point_step),),
        )

    def _predict(self, lidar_bev):
        value = normalize_lidar_bev(lidar_bev)
        tensor = torch.from_numpy(value).unsqueeze(0).to(self.device)
        probabilities = []
        with torch.no_grad():
            if self.mc_samples > 1:
                for module in self.model.modules():
                    if isinstance(module, torch.nn.Dropout2d):
                        module.train()
            for _ in range(self.mc_samples):
                probabilities.append(
                    torch.softmax(self.model(tensor), dim=1)[0, 1]
                )
            self.model.eval()
        stacked = torch.stack(probabilities)
        mean = stacked.mean(dim=0).cpu().numpy()
        variance = stacked.var(dim=0, unbiased=False).cpu().numpy()
        return mean, variance

    @staticmethod
    def _image_message(values, source, *, mask=False):
        array = np.asarray(values)
        output = Image()
        output.header = source.header
        output.height = int(array.shape[0])
        output.width = int(array.shape[1])
        output.is_bigendian = False
        if mask:
            stored = np.ascontiguousarray(array, dtype=np.uint8)
            output.encoding = 'mono8'
            output.step = int(output.width)
        else:
            stored = np.ascontiguousarray(array, dtype='<f4')
            output.encoding = '32FC1'
            output.step = int(output.width) * 4
        output.data = stored.tobytes()
        return output

    def _grid_message(self, values, source):
        converted = bev_to_occupancy_grid(values).astype(np.int8, copy=False)
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
            x = self._float32_view(message, 'x')
            y = self._float32_view(message, 'y')
            z = self._float32_view(message, 'z')
            points = np.column_stack((x, y, z))
            lidar_bev = build_lidar_bev(points, self.geometry)
            probability, variance = self._predict(lidar_bev)
            entropy = binary_predictive_entropy(probability)
            observed = lidar_bev[0] > 0.5
            decision = build_selective_decision(
                probability,
                entropy,
                observed,
                lidar_bev[2],
                lidar_bev[3],
                free_probability_threshold=float(self.get_parameter(
                    'free_probability_threshold'
                ).value),
                obstacle_probability_threshold=float(self.get_parameter(
                    'obstacle_probability_threshold'
                ).value),
                maximum_entropy=float(
                    self.get_parameter('maximum_entropy').value
                ),
                maximum_mc_variance=float(self.get_parameter(
                    'maximum_mc_variance'
                ).value),
                mc_variance=variance,
                hard_obstacle_minimum_z_m=float(self.get_parameter(
                    'hard_obstacle_minimum_z_m'
                ).value),
                hard_obstacle_vertical_span_m=float(self.get_parameter(
                    'hard_obstacle_vertical_span_m'
                ).value),
                exclusion_mask=self.ego_mask,
            )
        except (ValueError, RuntimeError) as error:
            self.get_logger().error('shadow inference failed: ' + str(error))
            return

        available = observed & ~self.ego_mask
        probability_grid = np.full(probability.shape, -1, dtype=np.int8)
        probability_grid[available] = np.rint(
            100.0 * probability[available]
        ).astype(np.int8)
        uncertainty_grid = np.full(probability.shape, -1, dtype=np.int8)
        uncertainty_grid[available] = np.rint(
            100.0 * entropy[available]
        ).astype(np.int8)
        self.probability_publisher.publish(
            self._grid_message(probability_grid, message)
        )
        self.uncertainty_publisher.publish(
            self._grid_message(uncertainty_grid, message)
        )
        self.decision_publisher.publish(
            self._grid_message(decision.decision, message)
        )
        if self.publish_diagnostic_images:
            self.diagnostic_publishers['probability'].publish(
                self._image_message(probability, message)
            )
            self.diagnostic_publishers['entropy'].publish(
                self._image_message(entropy, message)
            )
            self.diagnostic_publishers['variance'].publish(
                self._image_message(variance, message)
            )
            self.diagnostic_publishers['hard_mask'].publish(
                self._image_message(
                    decision.hard_obstacle_mask, message, mask=True
                )
            )

        self.frame_sequence += 1
        status = {
            'mode': 'shadow_no_control_authority',
            'checkpoint': str(self.checkpoint_path),
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
            'learned_free_cells': int(np.count_nonzero(
                decision.learned_free_mask
            )),
            'uncertain_cells': int(np.count_nonzero(
                decision.uncertain_mask
            )),
            'mean_entropy': (
                float(np.mean(entropy[available]))
                if np.any(available) else None
            ),
            'mean_mc_variance': (
                float(np.mean(variance[available]))
                if np.any(available) else None
            ),
            'publish_diagnostic_images': self.publish_diagnostic_images,
            'free_probability_threshold': float(self.get_parameter(
                'free_probability_threshold'
            ).value),
            'obstacle_probability_threshold': float(self.get_parameter(
                'obstacle_probability_threshold'
            ).value),
            'maximum_entropy': float(
                self.get_parameter('maximum_entropy').value
            ),
            'maximum_mc_variance': float(self.get_parameter(
                'maximum_mc_variance'
            ).value),
            'hard_obstacle_minimum_z_m': float(self.get_parameter(
                'hard_obstacle_minimum_z_m'
            ).value),
            'hard_obstacle_vertical_span_m': float(self.get_parameter(
                'hard_obstacle_vertical_span_m'
            ).value),
            'inference_ms': 1000.0 * (time.perf_counter() - started),
        }
        output = String()
        output.data = json.dumps(status, sort_keys=True)
        self.status_publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TraversabilityShadowNode()
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
