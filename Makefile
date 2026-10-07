export PYTHONPATH := src
.PHONY: install data train monitor serve mlflow-ui simulate simulate-drift test rollback docker
install:        ; pip install -r requirements.txt
data:           ; python -c "from prp.data import generate_dataset as g; g(20000,42).to_csv('data/raw/patients.csv',index=False)"
train:          ; python -m prp.flows train          # validate -> train -> gate -> promote (Prefect)
monitor:        ; python -m prp.flows monitor        # drift + performance + HITL health -> reports/
serve:          ; uvicorn prp.serve:app --host 0.0.0.0 --port 8000
mlflow-ui:      ; mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000
simulate:       ; python scripts/simulate_traffic.py --n 800
simulate-drift: ; python scripts/simulate_traffic.py --n 800 --drift
test:           ; pytest -q
rollback:       ; python -m prp.cli rollback
docker:         ; docker compose up --build
