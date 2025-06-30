# tune_lgbm.py
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from train_lgbm import produce_train_data, get_transformers, train_model_tune

if __name__ == "__main__":
    ray.init()

    data = produce_train_data()
    cat_trans, num_trans = get_transformers(data)

    search_space = {
        "num_leaves": tune.randint(20, 150),
        "learning_rate": tune.uniform(0.01, 0.3),
        "n_estimators": tune.randint(50, 200),
        "min_child_samples": tune.randint(10, 100),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
    }

    algo = OptunaSearch(metric="roc_auc", mode="max")

    analysis = tune.run(
        tune.with_parameters(train_model_tune, data=data, cat_trans=cat_trans, num_trans=num_trans),
        config=search_space,
        num_samples=30,
        search_alg=algo,
        resources_per_trial={"cpu": 2},
        name="lgbm_ray_tune"
    )

    print("Best config:", analysis.best_config)
    print("Best AUC:", analysis.best_result["roc_auc"])
