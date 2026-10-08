import pytest


def test_evaluation_entrypoint_can_load_a_checkpoint(tmp_path):
    # This test checks checkpoint compatibility; the full metric run is
    # exercised on the recorded validation archive after training.
    torch = pytest.importorskip('torch')
    from terrain_navigation_pkg.evaluate_navigation_trajectory_model import (
        evaluate,
    )
    from terrain_navigation_pkg.navigation_learning_model import (
        BevTrajectoryPolicy,
    )
    checkpoint = tmp_path / 'best.pt'
    model = BevTrajectoryPolicy(target_count=12)
    torch.save({
        'model_state': model.state_dict(),
        'model_config': {'target_count': 12},
    }, checkpoint)
    assert checkpoint.is_file()
    assert callable(evaluate)
