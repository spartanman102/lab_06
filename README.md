# Lab 06 – CI/CD dla modeli ML

## Struktura

```
Lab_06/
├── model.py              # train_and_predict(), get_accuracy()
├── main.py               # FastAPI app
├── test_model.py         # 4 testy jednostkowe (pytest)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .github/workflows/
    └── ci.yml            # GitHub Actions: testy + build Docker
```

## Zadanie 1 – Testy lokalne

```bash
pip install -r requirements.txt
pytest test_model.py -v
```

## Zadanie 2 – GitHub Actions

Push do gałęzi `main` automatycznie uruchamia testy w Actions.

## Zadanie 3 – Docker (automatyczny build po teście)

Obraz publikowany do `ghcr.io/<user>/<repo>/ml-iris-api:latest`.
