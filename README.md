## Setup ##

Before running any of the assignments, the required Python packages must be installed. All dependencies are listed in requirements.txt.

It is recommended to create and use a virtual environment:

1. Create a virtual environment:
    python -m venv venv
2. Activate the virtual environment:
    Windows:
    venv\Scripts\activate
    macOS / Linux:
    source venv/bin/activate
3. Install all required packages:
    pip install -r requirements.txt

After the installation is complete, the notebooks and Python applications can be executed normally.


## Task 1: Hyperparameter Evaluation and Data Augmentation

The experiments for Task 1 are documented in the Jupyter notebook parameters.ipynb.

The notebook contains:

- the tested data augmentation configurations,
- the training and validation results,
- plots of the learning curves,
- confusion matrices,
- prediction time measurements,
- a discussion of the approaches, assumptions, and results.

The notebook is structured using Markdown sections and includes all analyses required for the assignment.

## Task 2: Evaluation on a Custom Dataset

Custom gesture dataset is located in the my_dataset directory.

The dataset contains images for the following gesture classes:

- dislike
- like
- peace
- rock
- stop

The corresponding annotations are stored in annot-maximilian.json.

The trained classifier was evaluated on this custom dataset. The resulting confusion matrix is provided as conf_matrix.png.

## Task 3: Gesture-Controlled Camera Application

The camera application can be started directly via Python. Two optional command-line parameters are supported:

- --timer <int> specifies the selfie countdown timer in seconds (default: 5).
- --path <string> specifies the output path for the captured selfie (default: selfie.jpg).

After starting the application, a live camera preview is displayed. Hand detection is performed using MediaPipe. The required MediaPipe hand landmark model is included as hand_landmarker.task.

The application supports the following gesture controls:

- Like (thumbs up): Enable the sepia filter.
- Dislike (thumbs down): Disable the sepia filter.
- STOP sign: Start the selfie countdown. After the countdown reaches zero, the current image is captured and saved to the specified output path.
- PEACE: Toggle the zoom feature on and off.

To reduce false positives, gestures are only accepted if they are detected consistently for eight consecutive frames.

The gesture classification itself is performed using the trained CNN model stored in gesture_recognition.keras. The script train_model.py contains the code used to train this model.
The classifier was trained on more gesture classes than the application actually uses for actions. The idea was that the model might distinguish the relevant gestures more reliably if it also learns similar but unused gestures as separate classes, instead of forcing every hand pose into one of the control gestures

The application can be closed at any time by pressing the Escape key.
