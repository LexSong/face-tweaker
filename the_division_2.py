import os
import traceback

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntUniformDistribution
from optuna.samplers import TPESampler
from optuna.trial import Trial
from PIL import Image
from tqdm import tqdm

from utils.face_recognition import get_face_distance
from utils.face_recognition import get_face_encoding
from utils.imagefolder import ImageFolder
from utils.screenshot import get_screenshot


target_image = Image.open("data/target.jpg")
target_encoding = get_face_encoding(target_image)

screenshot_folder = ImageFolder("screenshots/the_division_2")


def params_to_filename(params: dict) -> str:
    return "_".join(str(x) for x in params.values())


def filename_to_params(filename) -> dict:
    values = filename.split("_")
    keys = [
        "Body Type",
        "Head",
        "Brow Height",
        "Brow Depth",
        "Eyeline",
        "Eye Spacing",
        "Nose Width",
        "Nose Height",
        "Nose Bridge",
        "Mouth Height",
        "Cheeks",
        "Jawline",
    ]
    params = dict()
    for k, v in zip(keys, values):
        if k == "Body Type":
            params[k] = v
        else:
            params[k] = int(v)
    return params


def print_params(params: dict) -> None:
    for k, v in params.items():
        if k in ["Body Type", "Head"]:
            print(f"{k}: {params[k]}")
            continue

        def bar_char(i):
            if i == v:
                return "O"
            if i % 3 == 0:
                return "."
            else:
                return "_"

        bar = "".join(bar_char(i) for i in range(13))
        print(k.rjust(15), bar, v)


def objective(trial: Trial):
    # trial.suggest_categorical("Body Type", ["Male", "Female"])
    trial.suggest_categorical("Body Type", ["Female"])
    trial.suggest_categorical("Head", range(20))
    trial.suggest_int("Brow Height", 0, 12)
    trial.suggest_int("Brow Depth", 0, 12)
    trial.suggest_int("Eyeline", 0, 12)
    trial.suggest_int("Eye Spacing", 0, 12)
    trial.suggest_int("Nose Width", 0, 12)
    trial.suggest_int("Nose Height", 0, 12)
    trial.suggest_int("Nose Bridge", 0, 12)
    trial.suggest_int("Mouth Height", 0, 12)
    trial.suggest_int("Cheeks", 0, 12)
    trial.suggest_int("Jawline", 0, 12)

    while True:
        try:
            print_params(trial.params)
            os.system("pause")
            screenshot = get_screenshot("Tom Clancy's The Division 2")

            # Save and load screenshot from the folder
            # Ensure get_face_encoding() can get the same results
            filename = params_to_filename(trial.params)
            screenshot_folder[filename] = screenshot
            screenshot = screenshot_folder[filename]

            encoding = get_face_encoding(screenshot)
            distance = get_face_distance(encoding, target_encoding)
            return distance
        except Exception:
            traceback.print_exc()


study = optuna.create_study(
    sampler=TPESampler(multivariate=True),
)

# Load existed screenshots
distributions = {
    "Body Type": CategoricalDistribution(choices=("Male", "Female")),
    "Head": CategoricalDistribution(choices=list(range(20))),
    "Brow Height": IntUniformDistribution(0, 12),
    "Brow Depth": IntUniformDistribution(0, 12),
    "Eyeline": IntUniformDistribution(0, 12),
    "Eye Spacing": IntUniformDistribution(0, 12),
    "Nose Width": IntUniformDistribution(0, 12),
    "Nose Height": IntUniformDistribution(0, 12),
    "Nose Bridge": IntUniformDistribution(0, 12),
    "Mouth Height": IntUniformDistribution(0, 12),
    "Cheeks": IntUniformDistribution(0, 12),
    "Jawline": IntUniformDistribution(0, 12),
}

for filename, screenshot in tqdm(screenshot_folder.items(), ascii=True):
    params = filename_to_params(filename)
    encoding = get_face_encoding(screenshot)
    distance = get_face_distance(encoding, target_encoding)
    trial = optuna.create_trial(params=params, distributions=distributions, value=distance)
    study.add_trial(trial)

study.optimize(objective, n_trials=10)

print()
print("Best settings:")
print_params(study.best_params)
