# Install Requirments
- conda install -c conda-forge graphviz python-graphviz -y
- dot -V
- pip install --upgrade pip setuptools wheel
- pip install -r requirements.txt
- pip install --no-deps git+https://github.com/SPFlow/SPFlow.git [if the code needs DeepDB/SPFlow and gives an error]

# Execute the code
```python udf_generator/create_db_statistics.py --dbms duckdb --dbms_kwargs dir=../Graceful_data/datasets/ --col_stats_dir ../Graceful_data/datasets/statistics/ --target ../Graceful_data/datasets/DB_metadata.csv```


tmux new -s est_training
bash run_paper_est.sh