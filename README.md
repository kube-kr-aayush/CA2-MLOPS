# Breast Cancer Classification — ML + MLOps

This project follows the instructor's required workflow:

1. **Google Colab:** train and evaluate the ML model.
2. Save and download **`model.joblib`** from Colab.
3. **VS Code:** load the saved model with FastAPI and test predictions.
4. Run the API through Docker.
5. Push to a new GitHub repository and verify GitHub Actions CI.

> Educational demonstration only. This model is not for medical diagnosis.

## Project layout

```text
app/main.py
models/model.joblib             # Put the file downloaded from Colab here
notebooks/colab_training.ipynb  # Run this in Google Colab, not VS Code
data/breast_cancer.csv          # Local reproducible dataset copy
tests/test_api.py
tests/sample_payload.json
requirements.txt
Dockerfile
.github/workflows/ci.yml
```

## Phase 1 — Train in Google Colab

1. Open `notebooks/colab_training.ipynb` in Google Colab (upload the notebook to Colab or open it from Google Drive).
2. Run all cells from top to bottom.
3. The notebook trains Logistic Regression, evaluates it on a held-out test set, and saves `model.joblib`.
4. Download `model.joblib` when the download prompt appears.
5. Create the `models` folder in your VS Code project if needed and place the downloaded file at `models/model.joblib`.
6. Download `sample_payload.json` from Colab and place it at `tests/sample_payload.json` if you want the matching sample for the trained dataset.

If your instructor specifically requires the Kaggle CSV: download the Wisconsin Diagnostic Breast Cancer CSV from Kaggle, upload it in Colab when prompted by setting `USE_UPLOADED_KAGGLE_CSV = True`, and run all cells. The notebook accepts common Kaggle column names.

## Phase 2 — Test in VS Code

Use Python 3.11 in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs. Use `POST /predict` to send a JSON payload and inspect the prediction. Check `/health` for model load status.

## Phase 3 — Docker

Ensure Docker Desktop is running, then:

```powershell
docker build -t breast-cancer-ml-api:1.0 .
docker run --rm -p 8000:8000 breast-cancer-ml-api:1.0
```

Open http://127.0.0.1:8000/docs.

## Phase 4 — GitHub repository

Create a new empty repository on GitHub. In the project folder:

```powershell
git init
git add .
git commit -m "Initial ML MLOps project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the example remote with your own repository URL.

## Phase 5 — GitHub Actions CI

After pushing, open your GitHub repository → Actions → **ML API CI**. The workflow installs dependencies and runs `pytest -q`. A green check confirms that the CI job passed.

## Dataset

Wisconsin Diagnostic Breast Cancer dataset, available on Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
