import pytest
import torch
import yaml
from click.testing import CliRunner
from scripts.pyrai_cli import cli
from pyrai.utils.model_utils import SimpleModel

@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()

@pytest.fixture
def model_path(tmp_path):
    """Create test model file."""
    model = SimpleModel()
    path = tmp_path / "model.pt"
    torch.save(model, path)
    return str(path)

def test_analyze(runner, model_path):
    """Test model analysis command."""
    result = runner.invoke(cli, [
        'analyze',
        model_path,
        '--input-size', '1,3,224,224'
    ])
    
    assert result.exit_code == 0
    assert "Model Analysis:" in result.output
    assert "Model Architecture:" in result.output

def test_export_onnx(runner, model_path, tmp_path):
    """Test ONNX export command."""
    output_path = tmp_path / "model.onnx"
    
    result = runner.invoke(cli, [
        'export',
        model_path,
        str(output_path),
        '--format', 'onnx',
        '--input-size', '1,3,224,224'
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()

def test_export_torchscript(runner, model_path, tmp_path):
    """Test TorchScript export command."""
    output_path = tmp_path / "model.pt"
    
    result = runner.invoke(cli, [
        'export',
        model_path,
        str(output_path),
        '--format', 'torchscript',
        '--input-size', '1,3,224,224'
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()

def test_benchmark(runner, model_path, tmp_path):
    """Test benchmark command."""
    result = runner.invoke(cli, [
        'benchmark',
        model_path,
        '--output-dir', str(tmp_path),
        '--batch-sizes', '1,2',
        '--iterations', '2'
    ])
    
    assert result.exit_code == 0
    results_path = tmp_path / "benchmark_results.yaml"
    assert results_path.exists()
    
    # Check results format
    with open(results_path) as f:
        results = yaml.safe_load(f)
        assert isinstance(results, dict)
        assert "1" in results  # batch size
        assert "latency" in results["1"] 