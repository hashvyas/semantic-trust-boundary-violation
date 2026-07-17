import importlib.util
import pathlib


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "b3_eval" / "run_model_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_model_benchmark", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_supports_seeds_argument():
    module = load_module()

    parser = module.build_arg_parser()
    args = parser.parse_args(["--seeds", "0", "1", "2", "3", "4"])

    assert args.seeds == [0, 1, 2, 3, 4]


def test_split_loader_falls_back_to_v1_dataset_json():
    module = load_module()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as tmp:
        tmp.write('{"text": "hello", "label": 1}\n')
        tmp_path = pathlib.Path(tmp.name)
    try:
        rows = module.load_jsonl(tmp_path)
        assert rows is not None
        assert len(rows) == 1
        assert rows[0] == ("hello", 1)
    finally:
        tmp_path.unlink()
