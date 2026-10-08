from terrain_navigation_pkg.gnss_goal_manager_node import (
    anchor_fix_is_new_enough,
)


def test_reanchor_rejects_fix_captured_before_odometry_jump():
    assert not anchor_fix_is_new_enough(
        latest_gnss_stamp_ns=9_900_000_000,
        minimum_anchor_gnss_stamp_ns=10_000_000_000,
    )


def test_reanchor_accepts_first_fix_at_or_after_odometry_jump():
    assert anchor_fix_is_new_enough(
        latest_gnss_stamp_ns=10_000_000_000,
        minimum_anchor_gnss_stamp_ns=10_000_000_000,
    )


def test_initial_anchor_only_requires_any_gnss_fix():
    assert anchor_fix_is_new_enough(
        latest_gnss_stamp_ns=1,
        minimum_anchor_gnss_stamp_ns=None,
    )
    assert not anchor_fix_is_new_enough(
        latest_gnss_stamp_ns=None,
        minimum_anchor_gnss_stamp_ns=None,
    )
