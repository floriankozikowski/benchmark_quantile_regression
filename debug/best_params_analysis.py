import pandas as pd

df = pd.read_parquet('benchopt_run_2025-07-03_19h12m20_allparams.parquet')

df_converged = df[df['stop_val'] == 0]

best_time = (
    df_converged
    .sort_values('time')
    .groupby('data_name', as_index=False)
    .first()
)
best_time.to_csv('top_by_time.csv', index=False)

best_obj = (
    df_converged
    .sort_values('objective_value')
    .groupby('data_name', as_index=False)
    .first()
)
best_obj.to_csv('top_by_objective.csv', index=False)

print('Exported top_by_time.csv and top_by_objective.csv.')
