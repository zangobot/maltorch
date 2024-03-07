import pathlib


class Config:
    MODEL_ZOO_FOLDER = pathlib.Path(__file__).parent / ".." / "zoo" / "models"
