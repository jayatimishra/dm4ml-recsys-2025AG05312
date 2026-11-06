<!-- Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment 2 -->
# models/ (Roll No: 2025AG05312)

`recommender_2025AG05312.pkl` is the released model (item-KNN collaborative filtering, retrained on
all data), versioned by DVC. The same model is logged to MLflow as a pyfunc model and registered as
`recsys_2025AG05312` with alias `champion`; the prediction service loads it from the registry.
Regenerate with `dvc repro train` or `python -m src.training.train`.
