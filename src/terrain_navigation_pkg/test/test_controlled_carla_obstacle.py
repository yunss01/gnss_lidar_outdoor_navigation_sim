import importlib.util
import math
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).parents[1]
    / 'terrain_navigation_pkg'
    / 'controlled_carla_obstacle.py'
)
SPEC = importlib.util.spec_from_file_location(
    'controlled_carla_obstacle_under_test', MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_planar_target_yaw_zero():
    x, y = MODULE.planar_target(10.0, 20.0, 0.0, 6.0, 2.0)
    assert math.isclose(x, 16.0)
    assert math.isclose(y, 22.0)


def test_planar_target_yaw_ninety():
    x, y = MODULE.planar_target(10.0, 20.0, 90.0, 6.0, 2.0)
    assert math.isclose(x, 8.0, abs_tol=1.0e-9)
    assert math.isclose(y, 26.0, abs_tol=1.0e-9)


def test_parser_defaults_to_low_box():
    args = MODULE._parser().parse_args(['spawn'])
    assert args.blueprint == 'static.prop.creasedbox01'
    assert args.forward_m == 6.0
    assert args.right_m == 0.0
