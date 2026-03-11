# Experiment Log

Use this file to record:
- run id
- config used
- dataset version
- model variant
- GPU / runtime
- in-domain metrics
- cross-dataset metrics
- cross-dataset scoring assumption
- mapped / ignored / unmatched classes
- notes / issues / next action

Suggested template:

```text
Run ID:
Config:
Checkpoint:
Dataset:
Model:
GPU / Runtime:

In-domain:
- split:
- AP50:
- AP50:95:
- Precision50:
- Recall50:

Cross-dataset:
- dataset:
- split:
- AP50:
- AP50:95:
- Precision50:
- Recall50:
- scoring assumption:

Class mapping:
- mapped classes:
- ignored classes:
- unmatched target classes:
- unmatched source classes:

Notes / issues:
Next action:
```
