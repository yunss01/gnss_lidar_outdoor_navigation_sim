import numpy as np
from sensor_msgs.msg import PointCloud2

from terrain_navigation_pkg.traversability_shadow_node import (
    TraversabilityShadowNode,
)


def test_float_diagnostic_image_preserves_values_and_stamp():
    source = PointCloud2()
    source.header.frame_id = 'lidar_frame'
    source.header.stamp.sec = 12
    source.header.stamp.nanosec = 34
    values = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    message = TraversabilityShadowNode._image_message(values, source)
    restored = np.frombuffer(message.data, dtype='<f4').reshape((2, 2))

    assert message.header == source.header
    assert message.encoding == '32FC1'
    assert message.step == 8
    assert np.allclose(restored, values)


def test_mask_diagnostic_image_is_packed_mono8():
    source = PointCloud2()
    values = np.asarray([[False, True], [True, False]])

    message = TraversabilityShadowNode._image_message(
        values, source, mask=True
    )
    restored = np.frombuffer(message.data, dtype=np.uint8).reshape((2, 2))

    assert message.encoding == 'mono8'
    assert message.step == 2
    assert restored.tolist() == [[0, 1], [1, 0]]
