<!-- Roll No: 2025AG05312 -->
# Model evaluation results (test window, K=10)

Evaluated on 921 users with at least one new add-to-cart or purchase in the 14-day test window. Selected model: **Item-KNN CF**.

| Model | Params | precision@10 | recall@10 | ndcg@10 | map@10 | hit_rate@10 | coverage | diversity | novelty | NDCG lift |
|---|---|---|---|---|---|---|---|---|---|---|
| Popularity (baseline) | `{}` | 0.0312 | 0.1312 | 0.0851 | 0.0526 | 0.2628 | 0.0683 | 0.8963 | 6.69 | +0.0% |
| **Item-KNN CF** | `{"neighbours": 100}` | 0.0451 | 0.1850 | 0.1244 | 0.0789 | 0.3529 | 0.4550 | 0.7191 | 7.53 | +46.1% |
| SVD matrix factorisation | `{"factors": 16}` | 0.0367 | 0.1504 | 0.0990 | 0.0605 | 0.2964 | 0.3817 | 0.6526 | 8.01 | +16.3% |
| Hybrid (SVD + content + popularity) | `{"factors": 32, "alpha": 0.25, "beta": 0.15}` | 0.0359 | 0.1505 | 0.1007 | 0.0624 | 0.2899 | 0.7100 | 0.3407 | 8.46 | +18.3% |

## Explicit rating prediction (supplementary)

| Model | RMSE | MAE | Test ratings |
|---|---|---|---|
| global_mean | 0.9688 | 0.7923 | 958 |
| bias_model | 0.9201 | 0.7587 | 958 |
