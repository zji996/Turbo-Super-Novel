from libs.pycore.paths import paths_summary


def test_paths_summary_shape() -> None:
    payload = paths_summary()
    assert {"models_dir", "data_dir", "logs_dir"} <= set(payload.keys())

