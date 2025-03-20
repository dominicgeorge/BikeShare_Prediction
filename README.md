
# Bike Rental Prediction Modular project

This project demonstrates a machine learning pipeline that processes data, builds a model, and saves it for future use.

## Setup

Follow the instructions below to set up and run the project locally.

### Prerequisites

Make sure you have the following installed:
- Python 3.x
- `pip` (Python's package installer)

### Installation

1. Clone this repository to your local machine.

2. Navigate to the project directory in your terminal.

3. Create a virtual environment:

   ```
   python -m venv application
   ```

### Activate the virtual environment:

#### On Windows

    .\application\Scripts\activate
    
#### On macOS/Linux

    source application/bin/activate

### Install the required dependencies

   ```
   cd .\requirements\ 
   pip install -r requirements.txt
   ```


   ```
   cd ..
   ```

### Running the Project

Once the environment is activated, run the main Python script:

    For Training
        cd .\bikeshare_model\ 
        python .\train_pipeline.py

        R2 score: 0.9931964108998564
        Mean squared error: 230.46862561373229
        Model/pipeline trained successfully!

    For Inference
        python .\predict.py
