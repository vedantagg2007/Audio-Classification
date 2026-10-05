import librosa
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader



SAMPLE_RATE = 16000

CROP_DURATION = 2
NUM_SAMPLES = SAMPLE_RATE * CROP_DURATION

TOP_K = 3

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "topk_data"

MODEL_PATH = BASE_DIR / "best_models/best_raw_topk_model.pth"

CLASS_NAMES = [
    "jackhammer",
    "siren",
    "chainsaw"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)


class AudioDataset(Dataset):

    def __init__(self, split):

        self.recording_folders = []
        self.labels = []

        print(f"\nLoading {split} recordings...")

        for label, class_name in enumerate(CLASS_NAMES):

            class_folder = (
                DATA_DIR
                / split
                / class_name
            )

            recording_folders = sorted(
                [
                    folder
                    for folder in class_folder.iterdir()
                    if folder.is_dir()
                ]
            )

            print(
                f"{class_name}: "
                f"{len(recording_folders)} recordings"
            )

            for folder in recording_folders:

                self.recording_folders.append(
                    folder
                )

                self.labels.append(
                    label
                )

    def __len__(self):

        return len(
            self.recording_folders
        )

    def __getitem__(self, index):

        folder = (
            self.recording_folders[index]
        )

        crops = []

        for crop_number in range(
            1,
            TOP_K + 1
        ):

            crop_path = (
                folder
                / f"crop_{crop_number}.wav"
            )

            if not crop_path.exists():

                raise FileNotFoundError(
                    f"Missing crop: {crop_path}"
                )

            audio, sr = librosa.load(
                crop_path,
                sr=SAMPLE_RATE,
                mono=True
            )

            # Guarantee exactly 2 seconds
            if len(audio) < NUM_SAMPLES:

                audio = np.pad(
                    audio,
                    (
                        0,
                        NUM_SAMPLES - len(audio)
                    )
                )

            else:

                audio = audio[
                    :NUM_SAMPLES
                ]

            audio = torch.tensor(
                audio,
                dtype=torch.float32
            ).unsqueeze(0)

            # Shape of one crop:
            # (1, 32000)

            crops.append(
                audio
            )

        # Shape:
        # (3, 1, 32000)

        crops = torch.stack(
            crops
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return crops, label



class AudioCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # First convolutional block
            nn.Conv1d(
                1, #1 input channel since mono
                16, #16 different filters
                kernel_size=15, #each filter looks at 15 different neighbors at a time
                stride=4, #moves forward by 4 after checking each sample
                padding=7 #adds 0's to end to stop data at edges from being discarded
            ),

            nn.ReLU(), #makes negative values 0

            nn.MaxPool1d(4), #Only keeps highest of 4 neighboring values to figure out if feature showed up strongly here

            # Second convolutional block
            nn.Conv1d(
                16,
                32, #32 new feature maps
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.MaxPool1d(4),

            # Third convolutional block
            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.MaxPool1d(4),

            # Reduce each feature map
            # to one value
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(
            64,
            len(CLASS_NAMES)
        )

    def forward(self, x):

        # Input shape:
        #
        # (batch, crops, channel, samples)
        #
        # Example:
        #
        # (8, 3, 1, 32000)

#8 original records per batch
#3 top k recordings per batch
#1 audio channel since mono
#32000 waveform signals per 2 seconds since 2*16khz


        batch_size = x.shape[0]
        num_crops = x.shape[1]

        # Merge batch and crop dimensions
        #
        # (8, 3, 1, 32000)
        #
        # becomes
        #
        # (24, 1, 32000)

        x = x.view(
            batch_size * num_crops,
            x.shape[2],
            x.shape[3]
        )

        # Run all crops through
        # the same CNN
        x = self.features(
            x
        )

        x = x.flatten(
            start_dim=1
        )

        logits = self.classifier(
            x
        )

        # Current shape:
        #
        # (batch * crops, classes)
        #
        # Example:
        #
        # (24, 3)
#24 crop rpedictions since 8 recordings times 3 crops each
#3 class scores per crop


        # Restore grouping
        #
        # (24, 3)
        #
        # becomes
        #
        # (8, 3, 3)
        #
        # recording
        # x crop
        # x class

        logits = logits.view(
            batch_size,
            num_crops,
            len(CLASS_NAMES)
        )

        # Average the logits
        # from the three crops
        #
        # Result:
        #
        # (batch, 3)

        logits = logits.mean(
            dim=1
        )

        return logits



def evaluate_model(
    model,
    loader
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for audio, labels in loader:

            audio = audio.to(
                device
            )

            outputs = model(
                audio
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    actual = np.array(
        all_labels
    )

    predicted = np.array(
        all_predictions
    )

    accuracy = np.mean(
        actual == predicted
    )

    return (
        accuracy,
        actual,
        predicted
    )




if __name__ == "__main__":

    print(
        "Using device:",
        device
    )



    train_dataset = AudioDataset(
        "train"
    )

    val_dataset = AudioDataset(
        "validation"
    )

    test_dataset = AudioDataset(
        "test"
    )

    if (
        len(train_dataset) == 0
        or len(val_dataset) == 0
        or len(test_dataset) == 0
    ):

        raise RuntimeError(
            "One or more dataset splits are empty."
        )



    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        "\nTraining recordings:",
        len(train_dataset)
    )

    print(
        "Validation recordings:",
        len(val_dataset)
    )

    print(
        "Testing recordings:",
        len(test_dataset)
    )


    audio, labels = next(
        iter(train_loader)
    )

    print(
        "\nBatch audio shape:",
        audio.shape
    )

    print(
        "Batch label shape:",
        labels.shape
    )

    # Should look approximately like:
    #
    # torch.Size([8, 3, 1, 32000])

    model = AudioCNN().to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float(
        "inf"
    )


    print(
        "\nStarting raw top-k training..."
    )

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0

        for audio, labels in train_loader:

            audio = audio.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            # One prediction per
            # original recording
            outputs = model(
                audio
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item()
                * len(labels)
            )

        train_loss = (
            total_train_loss
            / len(train_dataset)
        )



        model.eval()

        total_val_loss = 0
        val_correct = 0

        with torch.no_grad():

            for audio, labels in val_loader:

                audio = audio.to(
                    device
                )

                labels = labels.to(
                    device
                )

                outputs = model(
                    audio
                )

                loss = criterion(
                    outputs,
                    labels
                )

                total_val_loss += (
                    loss.item()
                    * len(labels)
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    predictions
                    == labels
                ).sum().item()

        val_loss = (
            total_val_loss
            / len(val_dataset)
        )

        val_accuracy = (
            val_correct
            / len(val_dataset)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2%}"
        )



        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

    print(
        "\nTraining complete!"
    )

    print(
        "Best model saved:",
        MODEL_PATH
    )


    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()


    accuracy, actual, predicted = (
        evaluate_model(
            model,
            test_loader
        )
    )

    print(
        f"\nTest Accuracy: "
        f"{accuracy:.2%}"
    )

    num_classes = len(
        CLASS_NAMES
    )

    cm = np.zeros(
        (
            num_classes,
            num_classes
        ),
        dtype=int
    )

    for actual_label, predicted_label in zip(
        actual,
        predicted
    ):

        cm[
            actual_label,
            predicted_label
        ] += 1

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        "\nClass order:",
        CLASS_NAMES
    )


    print(
        "\nClassification Report:"
    )

    print(
        f"{'Class':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
        f"{'Support':>10}"
    )

    f1_scores = []

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        # True positives
        tp = cm[i, i]

        # False positives
        fp = (
            cm[:, i].sum()
            - tp
        )

        # False negatives
        fn = (
            cm[i, :].sum()
            - tp
        )

        # Actual recordings
        # belonging to this class
        support = (
            cm[i, :].sum()
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if precision + recall > 0
            else 0
        )

        f1_scores.append(
            f1
        )

        print(
            f"{class_name:<15}"
            f"{precision:>12.2f}"
            f"{recall:>12.2f}"
            f"{f1:>12.2f}"
            f"{support:>10}"
        )


    macro_f1 = np.mean(
        f1_scores
    )

    print(
        f"\nMacro F1-score: "
        f"{macro_f1:.4f}"
    )

