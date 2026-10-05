#removes recordings that belong to more than one of the three classes from filtered_audio and copies the remaining files into cnn_data
from pathlib import Path
from collections import defaultdict
import shutil

BASE_DIR = Path(__file__).resolve().parent

source = BASE_DIR / "filtered_audio"
destination = BASE_DIR / "cnn_data"

classes = ["jackhammer", "siren", "chainsaw"]
splits = ["train", "validation", "test"]

# Track which classes each recording belongs to
recordings = defaultdict(set)

for split in splits:
    for label in classes:
        folder = source / split / label

        for path in folder.glob("*.wav"):
            recordings[path.name].add(label)

# Copy recordings with exactly one target label
counts = defaultdict(int)
excluded = 0

for split in splits:
    for label in classes:
        folder = source / split / label

        for path in folder.glob("*.wav"):

            # Exclude recordings with multiple labels
            if len(recordings[path.name]) != 1:
                excluded += 1
                continue

            output = destination / split / label

            output.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                path,
                output / path.name
            )

            counts[(split, label)] += 1

# Print results
for split in splits:
    print(f"\n{split.upper()}")

    for label in classes:
        print(f"{label}: {counts[(split, label)]}")

print(f"\nExcluded file occurrences: {excluded}")
