"""
Production-level Client Simulator.

Sends realistic HTTP requests to the deployed FastAPI container to generate
Prometheus metrics, monitors drift actively per batch, retrains the model
using MLflow tracking server locally if policy conditions meet, and live-swaps 
the model in the container via /reload-model without container restarts.
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

from src.utils import get_path
from src.drift_detection import DriftDetector
from src.policy_engine import RetrainingPolicyEngine
from src.retrain import retrain_model, ModelVersionManager
from src.feature_encoding import EncoderPipeline

API_URL = "http://localhost:8080"

def wait_for_api():
    print("Waiting for Inference API to be ready...")
    while True:
        try:
            r = requests.get(f"{API_URL}/health")
            if r.status_code == 200:
                print("API is up and healthy!")
                break
        except requests.exceptions.ConnectionError:
            time.sleep(2)

def simulate_production_traffic(batch_df: pd.DataFrame, batch_name: str) -> None:
    """Send data to API row-by-row to simulate realistic traffic & populate Grafana."""
    print(f"Simulating traffic for {batch_name}...")
    headers = {"Content-Type": "application/json"}
    
    # We send in chunks of 50 to stress the API but remain relatively fast
    chunk_size = 50
    for i in tqdm(range(0, len(batch_df), chunk_size), desc="Traffic"):
        chunk = batch_df.iloc[i:i+chunk_size].to_dict(orient="records")
        payload = {"records": chunk}
        
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, headers=headers)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"HTTP Error: {e}")

def run_simulation(policy_config: dict):
    wait_for_api()
    os.makedirs("mlruns", exist_ok=True)
    
    # 1. Base Setup
    train_data = pd.read_csv(get_path("processed_data_dir") / "train.csv")
    detector = DriftDetector(train_data)
    policy_engine = RetrainingPolicyEngine({"retraining_policy": policy_config})
    version_mgr = ModelVersionManager()
    
    # Current state of the encoding to avoid errors
    encoder = EncoderPipeline.load(get_path("models_dir") / "encoder_pipeline.pkl")
    
    batches = {}
    for i in range(1, 6):
        bpath = get_path("batch_data_dir") / f"batch_{i}.csv"
        if bpath.exists():
            batches[f"batch_{i}"] = pd.read_csv(bpath)

    acc_samples = pd.DataFrame()

    # 2. Iterate batches representing time
    for batch_name, batch_df in batches.items():
        print(f"\n{'='*50}\nProcessing {batch_name}\n{'='*50}")
        
        # Simulating external incoming REST traffic -> triggers Prometheus gauges
        simulate_production_traffic(batch_df, batch_name)
        
        # Drift Detection
        drift_res = detector.detect(batch_df, batch_name)
        acc_samples = pd.concat([acc_samples, batch_df])
        
        # Policy evaluation
        decision = policy_engine.evaluate(drift_res, len(acc_samples))
        
        retrain_triggered = False
        print(f"Drift Severity: {drift_res.severity_score:.4f} | Threshold: {policy_config.get('severity_threshold')}")
        
        if decision.should_retrain:
            print(f"*** RETRAINING TRIGGERED for {batch_name} ***")
            retrain_triggered = True
            
            # Retrain natively (will log automatically to MLflow at http://localhost:5000)
            ret_res = retrain_model(
                acc_samples, encoder, version_mgr, 
                old_train_data=train_data, use_mlflow=True
            )
            
            # Request zero-downtime hot swap to inference API
            print("Sending /reload-model signal to API...")
            r_reload = requests.post(f"{API_URL}/reload-model")
            if r_reload.status_code == 200:
                print(f"API successfully reloaded to version '{ret_res['version']}'!")
            
            # Update reference distributions for future monitoring
            train_data = pd.concat([train_data, acc_samples])
            acc_samples = pd.DataFrame()
            detector.update_reference(train_data)

        policy_engine.register_batch_processed(retrain_triggered)
        
        # Explicit call to API to record drift metrics to Prometheus
        try:
            requests.post(
                f"{API_URL}/internal/update-drift-metrics",
                params={"severity": drift_res.severity_score, "retrain_triggered": retrain_triggered}
            )
        except Exception as e:
            print(f"Could not push drift metrics to API: {e}")
            
    print("\nSimulation complete. Please check the Grafana dashboard on localhost:3000 to review live metrics!")
    print("Please check MLflow on localhost:5000 to review recorded parameters, model metrics and artifacts!")

if __name__ == "__main__":
    standard_policy = {
        "severity_threshold": 0.15,
        "min_new_samples": 1000,
        "cooldown_batches": 1
    }
    run_simulation(standard_policy)
