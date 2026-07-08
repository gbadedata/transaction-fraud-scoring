"""Transaction fraud scoring & alert-prioritisation engine.

A leakage-safe, capacity-aware fraud scoring pipeline:
    data  ->  features  ->  rules + model + anomaly  ->  blended score
          ->  expected-loss ranking  ->  capacity-tuned threshold  ->  monitoring

Every design choice here is deliberate for a fraud context; see docs/model_card.md
and docs/decisions/ for the reasoning.
"""

from fraud import data, features, metrics, model, monitoring, rules, scoring

__all__ = ["data", "features", "metrics", "model", "monitoring", "rules", "scoring"]
__version__ = "0.1.0"
