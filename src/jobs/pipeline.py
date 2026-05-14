import json
import logging
import time
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_pipeline_config(config_path: str | Path) -> dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_required_outputs(required_outputs: list[str]) -> dict:
    """Check if all required output files exist."""
    status = {"all_present": True, "outputs": []}
    for output_path in required_outputs:
        exists = Path(output_path).exists()
        status["outputs"].append({"path": output_path, "exists": exists})
        if not exists:
            status["all_present"] = False
    return status


def run_pipeline_step(step_name: str, command_name: str) -> dict:
    """
    Run a single pipeline step by mapping its command name to the corresponding internal function.
    """
    start_time = datetime.now()
    result = {
        "step_name": step_name,
        "command": command_name,
        "started_at": start_time.isoformat(),
        "status": "success",
        "error": None,
    }

    try:
        from src.data.download import download_movielens
        from src.data.load_raw import load_raw_tables, load_data_config
        from src.data.validate_schema import validate_raw_tables
        from src.data.prepare_interactions import run_prepare_interactions
        from src.features.build_features import run_build_features
        from src.retrieval.build_candidates import run_build_popularity_candidates
        from src.retrieval.build_itemknn_candidates import run_build_itemknn_candidates
        from src.evaluation.run_retrieval_evaluation import run_retrieval_evaluation
        from src.ranking.build_ranking_dataset import run_build_ranking_datasets
        from src.ranking.run_train_ranker import run_train_ranker

        # Map command names to functions (and default configs)
        # This avoids subprocess and uses internal Python calls
        dispatch = {
            "download-data": lambda: download_movielens(Path("configs/data.yaml")),
            "validate-raw": lambda: validate_raw_tables(load_raw_tables(Path("configs/data.yaml")), load_data_config(Path("configs/data.yaml"))),
            "prepare-interactions": lambda: run_prepare_interactions(Path("configs/data.yaml"), Path("configs/interactions.yaml")),
            "build-features": lambda: run_build_features(Path("configs/data.yaml"), Path("configs/features.yaml")),
            "build-popularity-candidates": lambda: run_build_popularity_candidates(Path("configs/retrieval.yaml")),
            "build-itemknn-candidates": lambda: run_build_itemknn_candidates(Path("configs/retrieval.yaml")),
            "evaluate-retrieval": lambda: run_retrieval_evaluation(Path("configs/evaluation.yaml")),
            "build-ranking-dataset": lambda: run_build_ranking_datasets(Path("configs/ranking.yaml")),
            "train-ranker": lambda: run_train_ranker(Path("configs/ranker.yaml")),
        }

        if command_name not in dispatch:
            raise ValueError(f"Unknown command: {command_name}")
        
        logger.info(f"Running step: {step_name} ({command_name})...")
        dispatch[command_name]()
    except Exception as e:
        logger.exception(f"Step {step_name} failed: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
    
    end_time = datetime.now()
    result["finished_at"] = end_time.isoformat()
    result["duration_seconds"] = (end_time - start_time).total_seconds()
    
    return result


def save_pipeline_run_metadata(
    metadata: dict, output_dir: str | Path, latest_status_path: str | Path
) -> None:
    """Save run metadata to a timestamped file and update latest_status.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file = output_dir / f"run_{timestamp}.json"
    
    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    latest_status_path = Path(latest_status_path)
    latest_status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latest_status_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def get_git_commit() -> str | None:
    """Attempt to get current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return None


def run_full_pipeline(
    pipeline_config_path: str | Path = "configs/pipeline.yaml",
    skip_existing_outputs: bool | None = None,
) -> dict:
    """Run all pipeline steps defined in the config."""
    config = load_pipeline_config(pipeline_config_path)["pipeline"]
    
    run_metadata = {
        "pipeline_name": config["name"],
        "pipeline_version": config["version"],
        "started_at": datetime.now().isoformat(),
        "status": "success",
        "steps": [],
        "failed_step": None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(),
    }
    
    stop_on_failure = config["behavior"]["stop_on_failure"]
    if skip_existing_outputs is None:
        skip_existing_outputs = config["behavior"]["skip_existing_outputs"]

    start_time = time.time()
    
    for i, step in enumerate(config["steps"]):
        step_name = step["name"]
        command = step["command"]
        required_outputs = step.get("required_outputs", [])
        
        # Check if we should skip
        if skip_existing_outputs and required_outputs:
            out_status = check_required_outputs(required_outputs)
            if out_status["all_present"]:
                logger.info(f"Skipping step {step_name} as outputs already exist.")
                run_metadata["steps"].append({
                    "step_name": step_name,
                    "status": "skipped",
                    "reason": "outputs_exist"
                })
                continue

        step_result = run_pipeline_step(step_name, command)
        
        # Check outputs after run
        if step_result["status"] == "success" and required_outputs:
            out_status = check_required_outputs(required_outputs)
            step_result["required_outputs_status"] = out_status
            if not out_status["all_present"]:
                step_result["status"] = "failed"
                step_result["error"] = "Required outputs missing after execution."
        else:
            step_result["required_outputs_status"] = {"all_present": False, "outputs": []}

        run_metadata["steps"].append(step_result)
        
        if step_result["status"] == "failed":
            run_metadata["status"] = "failed"
            run_metadata["failed_step"] = step_name
            if stop_on_failure:
                break
                
    run_metadata["finished_at"] = datetime.now().isoformat()
    run_metadata["duration_seconds"] = time.time() - start_time
    
    if config["behavior"]["write_run_metadata"]:
        save_pipeline_run_metadata(
            run_metadata,
            config["output"]["run_metadata_dir"],
            config["output"]["latest_status_path"]
        )
        
    return run_metadata


def check_pipeline_status(
    pipeline_config_path: str | Path = "configs/pipeline.yaml",
) -> dict:
    """Check required outputs for all steps without running them."""
    config = load_pipeline_config(pipeline_config_path)["pipeline"]
    
    status = {
        "pipeline_name": config["name"],
        "steps": []
    }
    
    for step in config["steps"]:
        out_status = check_required_outputs(step.get("required_outputs", []))
        status["steps"].append({
            "step_name": step["name"],
            "outputs_present": out_status["all_present"],
            "missing_outputs": [o["path"] for o in out_status["outputs"] if not o["exists"]],
            "status": "ready" if out_status["all_present"] else "incomplete"
        })
        
    return status
