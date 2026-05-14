import json
import pytest
from pathlib import Path
from src.jobs.pipeline import (
    load_pipeline_config,
    check_required_outputs,
    run_full_pipeline,
    check_pipeline_status,
    save_pipeline_run_metadata
)

@pytest.fixture
def mock_pipeline_config(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "pipeline.yaml"
    
    content = {
        "pipeline": {
            "name": "test_pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "step1",
                    "command": "cmd1",
                    "required_outputs": [str(tmp_path / "out1.txt")]
                },
                {
                    "name": "step2",
                    "command": "cmd2",
                    "required_outputs": [str(tmp_path / "out2.txt")]
                }
            ],
            "output": {
                "run_metadata_dir": str(tmp_path / "runs"),
                "latest_status_path": str(tmp_path / "latest.json")
            },
            "behavior": {
                "stop_on_failure": True,
                "skip_existing_outputs": False,
                "write_run_metadata": True
            }
        }
    }
    
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(content, f)
    
    return config_path

def test_load_pipeline_config_success(mock_pipeline_config):
    config = load_pipeline_config(mock_pipeline_config)
    assert config["pipeline"]["name"] == "test_pipeline"

def test_check_required_outputs_all_present(tmp_path):
    f1 = tmp_path / "file1.txt"
    f1.write_text("hello")
    
    status = check_required_outputs([str(f1)])
    assert status["all_present"] is True
    assert status["outputs"][0]["exists"] is True

def test_check_required_outputs_missing_files(tmp_path):
    f1 = tmp_path / "missing.txt"
    
    status = check_required_outputs([str(f1)])
    assert status["all_present"] is False
    assert status["outputs"][0]["exists"] is False

def test_pipeline_status_reports_missing_outputs(mock_pipeline_config):
    status = check_pipeline_status(mock_pipeline_config)
    assert status["pipeline_name"] == "test_pipeline"
    assert len(status["steps"]) == 2
    assert status["steps"][0]["outputs_present"] is False
    assert "out1.txt" in status["steps"][0]["missing_outputs"][0]

def test_save_pipeline_run_metadata_creates_files(tmp_path):
    metadata = {"status": "success"}
    run_dir = tmp_path / "runs"
    latest_path = tmp_path / "latest.json"
    
    save_pipeline_run_metadata(metadata, run_dir, latest_path)
    
    assert latest_path.exists()
    runs = list(run_dir.glob("run_*.json"))
    assert len(runs) == 1

def test_run_full_pipeline_stops_on_failure(mock_pipeline_config, monkeypatch, tmp_path):
    # Mock run_pipeline_step to fail on first step
    def mock_run_step(step_name, command):
        return {
            "step_name": step_name,
            "status": "failed",
            "error": "forced failure"
        }
    
    monkeypatch.setattr("src.jobs.pipeline.run_pipeline_step", mock_run_step)
    
    res = run_full_pipeline(mock_pipeline_config)
    
    assert res["status"] == "failed"
    assert res["failed_step"] == "step1"
    assert len(res["steps"]) == 1 # Stopped after first failure

def test_run_full_pipeline_success_with_mocked_steps(mock_pipeline_config, monkeypatch, tmp_path):
    # Create outputs so check_required_outputs passes
    (tmp_path / "out1.txt").write_text("done")
    (tmp_path / "out2.txt").write_text("done")
    
    def mock_run_step(step_name, command):
        return {
            "step_name": step_name,
            "status": "success",
            "error": None
        }
    
    monkeypatch.setattr("src.jobs.pipeline.run_pipeline_step", mock_run_step)
    
    res = run_full_pipeline(mock_pipeline_config)
    
    assert res["status"] == "success"
    assert len(res["steps"]) == 2
    assert res["steps"][0]["status"] == "success"
    assert res["steps"][1]["status"] == "success"

def test_run_full_pipeline_writes_metadata_on_failure(mock_pipeline_config, monkeypatch, tmp_path):
    # This test verifies that if a step fails (e.g., due to an ImportError or any other exception),
    # the orchestrator catches it, marks the run as failed, and STILL writes the metadata.
    def mock_run_step(step_name, command):
        return {
            "step_name": step_name,
            "status": "failed",
            "error": "simulated import or runtime error",
            "command": command
        }
    
    monkeypatch.setattr("src.jobs.pipeline.run_pipeline_step", mock_run_step)
    
    res = run_full_pipeline(mock_pipeline_config)
    
    # Assert orchestrator logic
    assert res["status"] == "failed"
    assert res["failed_step"] == "step1"
    
    # Check that latest_status.json was actually written
    import yaml
    with open(mock_pipeline_config, "r") as f:
        cfg = yaml.safe_load(f)["pipeline"]
    latest_status_path = Path(cfg["output"]["latest_status_path"])
    
    assert latest_status_path.exists()
    with open(latest_status_path, "r") as f:
        import json
        metadata = json.load(f)
        
    assert metadata["status"] == "failed"
    assert metadata["failed_step"] == "step1"
    assert len(metadata["steps"]) == 1

def test_pipeline_command_mapping_imports_are_valid():
    """Verify that all commands in the dispatch mapping can be imported without error."""
    # By calling run_pipeline_step with an unknown command, we execute all the imports
    # at the top of the try block. If any import is invalid (like the previous load_data_config issue),
    # it will be caught and returned as a failed step. If imports are valid, it will return
    # failed but with a 'Unknown command' ValueError instead.
    from src.jobs.pipeline import run_pipeline_step
    
    res = run_pipeline_step("test_step", "non-existent-command")
    assert res["status"] == "failed"
    assert "Unknown command" in res["error"]
