<!-- Roll No: 2025AG05312 -->
# Model evaluation results (test window, K=10)

Evaluated on 853 users with at least one new add-to-cart or purchase in the 14-day test window. Selected model: **Item-KNN CF**.

| Model | Params | precision@10 | recall@10 | ndcg@10 | map@10 | hit_rate@10 | coverage | diversity | novelty | NDCG lift |
|---|---|---|---|---|---|---|---|---|---|---|
| Popularity (baseline) | `{}` | 0.0315 | 0.1468 | 0.0951 | 0.0604 | 0.2638 | 0.0633 | 0.8818 | 6.62 | +0.0% |
| **Item-KNN CF** | `{"neighbours": 200}` | 0.0417 | 0.1911 | 0.1217 | 0.0766 | 0.3435 | 0.5700 | 0.7086 | 7.60 | +28.0% |
| SVD matrix factorisation | `{"factors": 16}` | 0.0345 | 0.1548 | 0.0974 | 0.0600 | 0.2837 | 0.3700 | 0.6791 | 7.94 | +2.4% |
| Hybrid (SVD + content + popularity) | `{"factors": 32, "alpha": 0.1, "beta": 0.15}` | 0.0329 | 0.1482 | 0.0908 | 0.0537 | 0.2814 | 0.5717 | 0.5880 | 8.12 | -4.5% |

## Explicit rating prediction (supplementary)

| Model | RMSE | MAE | Test ratings |
|---|---|---|---|
| global_mean | 0.9484 | 0.7470 | 943 |
| bias_model | 0.8984 | 0.7212 | 943 |
