Run queues as they were launched (one file per batch of runs, named by date): they record which configs,
seeds and folders produced the results under train/step1B, and can be started again to resume (finished
runs are skipped). The reusable pieces are in scripts/: seed_study.sh, train_folds.sh, evaluate_folds.sh.
