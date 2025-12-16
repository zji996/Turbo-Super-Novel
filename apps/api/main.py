from fastapi import FastAPI

from libs.pycore.paths import paths_summary

app = FastAPI(title="Turbo-Super-Novel API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/paths")
def paths() -> dict:
    return paths_summary()

