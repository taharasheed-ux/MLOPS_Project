"""
Tests for the retrain policy engine.
"""

import pytest
from src.policy_engine import RetrainingPolicyEngine
from src.drift_detection import BatchDriftResult

@pytest.fixture
def config():
    return {
        "retraining_policy": {
            "severity_threshold": 0.3,
            "min_new_samples": 500,
            "cooldown_batches": 2
        }
    }

@pytest.fixture
def drift_no_alert():
    return BatchDriftResult(
        batch_name="batch_1",
        feature_results=[],
        n_features_tested=10,
        n_features_drifted=0,
        severity_score=0.1,  # < threshold
        drift_detected=False
    )

@pytest.fixture
def drift_alert():
    return BatchDriftResult(
        batch_name="batch_2",
        feature_results=[],
        n_features_tested=10,
        n_features_drifted=4,
        severity_score=0.5,  # > threshold
        drift_detected=True
    )

def test_engine_initialization(config):
    engine = RetrainingPolicyEngine(config)
    assert engine.severity_threshold == 0.3
    assert engine.min_new_samples == 500
    assert engine.cooldown_batches == 2
    # Should start ready to retrain
    assert engine.batches_since_retrain == 2 

def test_all_conditions_met(config, drift_alert):
    engine = RetrainingPolicyEngine(config)
    decision = engine.evaluate(drift_alert, current_new_samples=600)
    
    assert decision.should_retrain is True
    assert decision.severity_met is True
    assert decision.samples_met is True
    assert decision.cooldown_met is True

def test_severity_not_met(config, drift_no_alert):
    engine = RetrainingPolicyEngine(config)
    decision = engine.evaluate(drift_no_alert, current_new_samples=600)
    
    assert decision.should_retrain is False
    assert decision.severity_met is False

def test_samples_not_met(config, drift_alert):
    engine = RetrainingPolicyEngine(config)
    decision = engine.evaluate(drift_alert, current_new_samples=200) # < 500
    
    assert decision.should_retrain is False
    assert decision.samples_met is False

def test_cooldown_not_met(config, drift_alert):
    engine = RetrainingPolicyEngine(config)
    # Simulate a recent retrain
    engine.batches_since_retrain = 1 # Cooldown is 2
    decision = engine.evaluate(drift_alert, current_new_samples=600)
    
    assert decision.should_retrain is False
    assert decision.cooldown_met is False

def test_state_management(config, drift_alert):
    engine = RetrainingPolicyEngine(config)
    
    # 1. Evaluate & trigger retrain
    decision = engine.evaluate(drift_alert, 600)
    assert decision.should_retrain is True
    
    # Register retrain
    engine.register_batch_processed(retrain_triggered=True)
    assert engine.batches_since_retrain == 0
    
    # 2. Next batch, even if severe drift, cooldown blocks
    decision_2 = engine.evaluate(drift_alert, 600)
    assert decision_2.should_retrain is False
    assert decision_2.cooldown_met is False
    
    # Register no retrain
    engine.register_batch_processed(retrain_triggered=False)
    assert engine.batches_since_retrain == 1

    # 3. Next batch, cooldown STILL blocks (needs >= 2)
    decision_3 = engine.evaluate(drift_alert, 600)
    assert decision_3.should_retrain is False
    
    engine.register_batch_processed(retrain_triggered=False)
    assert engine.batches_since_retrain == 2
    
    # 4. Next batch, cooldown satisfied
    decision_4 = engine.evaluate(drift_alert, 600)
    assert decision_4.should_retrain is True
