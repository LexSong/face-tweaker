import os
import traceback

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.samplers import RandomSampler
from optuna.trial import Trial
from PIL import Image
from tqdm import tqdm

from utils.face_recognition import get_face_distance
from utils.face_recognition import get_face_encoding
from utils.imagefolder import ImageFolder
from utils.screenshot import get_screenshot


target_image = Image.open("data/target.jpg")
target_encoding = get_face_encoding(target_image)

screenshot_folder = ImageFolder("screenshots/cyberpunk2077")


def params_to_filename(params: dict) -> str:
    return "_".join(str(x) for x in params.values())


def filename_to_params(filename) -> dict:
    values = filename.split("_")
    keys = [
        "Body Type",
        "Skin Tone",
        "Skin Type",
        "Eyes",
        "Eyebrows",
        "Nose",
        "Mouth",
        "Jaw",
        "Ears",
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
        print(f"{k}: {v}")


def choices(n: int):
    return list(range(1, n + 1))


def objective(trial: Trial):
    # trial.suggest_categorical("Body Type", ["Male", "Female"])
    trial.suggest_categorical("Body Type", ["Female"])
    trial.suggest_categorical("Skin Tone", choices(12))
    trial.suggest_categorical("Skin Type", choices(5))
    trial.suggest_categorical("Eyes", choices(21))
    trial.suggest_categorical("Eyebrows", choices(8))
    trial.suggest_categorical("Nose", choices(21))
    trial.suggest_categorical("Mouth", choices(21))
    trial.suggest_categorical("Jaw", choices(21))
    trial.suggest_categorical("Ears", choices(21))

    while True:
        try:
            print_params(trial.params)
            os.system("pause")
            screenshot = get_screenshot("Cyberpunk 2077")

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


# study = optuna.create_study(sampler=RandomSampler())
study = optuna.create_study()

# Load existed screenshots
distributions = {
    "Body Type": CategoricalDistribution(choices=("Male", "Female")),
    "Skin Tone": CategoricalDistribution(choices=choices(12)),
    "Skin Type": CategoricalDistribution(choices=choices(5)),
    "Eyes": CategoricalDistribution(choices=choices(21)),
    "Eyebrows": CategoricalDistribution(choices=choices(8)),
    "Nose": CategoricalDistribution(choices=choices(21)),
    "Mouth": CategoricalDistribution(choices=choices(21)),
    "Jaw": CategoricalDistribution(choices=choices(21)),
    "Ears": CategoricalDistribution(choices=choices(21)),
}

for filename, screenshot in tqdm(screenshot_folder.items(), ascii=True):
    params = filename_to_params(filename)
    encoding = get_face_encoding(screenshot)
    distance = get_face_distance(encoding, target_encoding)
    trial = optuna.create_trial(
        params=params, distributions=distributions, value=distance
    )
    study.add_trial(trial)

study.optimize(objective, n_trials=5)

print()
print("Best settings:")
print_params(study.best_params)
