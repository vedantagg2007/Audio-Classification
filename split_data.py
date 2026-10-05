#Makes it so that I'm picking x amount for training y for validation z for testing since originally with dataset labels it wasn't balanced
from pathlib import Path
from collections import Counter
import random
import shutil
import csv

# 1. Set up directories
BASE_DIR = Path(__file__).resolve().parent

source = BASE_DIR / "cnn_data"
destination = BASE_DIR / "balanced_data"

classes = ["jackhammer", "siren", "chainsaw"]
splits = ["train", "validation", "test"]

# 2. Define how many recordings we want
TRAIN_SIZE = 400
VAL_SIZE = 75
TEST_SIZE = 75

TOTAL = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

# Choosing seed makes the random selection reproducible
random.seed(42)

# Prevent mixing old and new results
if destination.exists():
    raise FileExistsError(
        "balanced_data already exists. "
        "Rename or remove it before running again."
    )

# 3. Collect all recordings
all_files = {}
filename_classes = {}

for label in classes:
    recordings = []

    for split in splits:
        folder = source / split / label

        for path in folder.glob("*.wav"):
            # Detect duplicate recordings
            if path.name in filename_classes:
                raise ValueError(
                    f"Duplicate filename: {path.name}"
                )

            filename_classes[path.name] = label
            recordings.append((path, split))

    all_files[label] = recordings

# 4. Check that we have enough recordings
for label in classes:
    available = len(all_files[label])

    print(f"{label}: {available} available")

    if available < TOTAL:
        raise ValueError(
            f"Not enough {label} recordings. "
            f"Need {TOTAL}, have {available}."
        )

# 5. Randomly split each class
counts = Counter()
manifest = []

for label in classes:

    recordings = all_files[label].copy()

    # Randomize the order
    random.shuffle(recordings)

    # Select recordings for each split
    train = recordings[:TRAIN_SIZE]

    validation = recordings[
        TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE
    ]

    test = recordings[
        TRAIN_SIZE + VAL_SIZE:TOTAL
    ]

    selected = {
        "train": train,
        "validation": validation,
        "test": test
    }

    # 6. Copy recordings into their new folders
    for split, files in selected.items():

        folder = destination / split / label
        folder.mkdir(parents=True, exist_ok=True)

        for path, original_split in files:

            shutil.copy2(
                path,
                folder / path.name
            )

            manifest.append([
                path.name,
                label,
                original_split,
                split
            ])

            counts[(split, label)] += 1

# 7. Save a record of the new assignments
with open(
    destination / "split_manifest.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "filename",
        "class",
        "original_split",
        "new_split"
    ])

    writer.writerows(manifest)

# 8. Print the final counts
for split in splits:

    print(f"\n{split.upper()}")

    for label in classes:
        print(f"{label}: {counts[(split, label)]}")

print("\nDataset preparation complete!")
