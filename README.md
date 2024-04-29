<div align="center">
    <h1 align="center">NLP Based Agents in Video Games</h1>
</div>

## Created By
* Chris Pitre
* Samith Shetty

## Special Thanks To
* PyTorch
* HuggingFace
* OpenAI

## Instructions

### Training the Model
Training the Model is relatively easy, but requires a lot of processing power (namely a NVIDIA GPU with CUDA)
Steps:
1. Install prerequisite modules
```sh
pip install torch pandas numpy sklearn transformers matplotlib
```
2. Run model.py
3. Select 1 to train generation or 0 to train classification

### Running the Game
To run the full game as intended, you will need to have Godot 4.2 installed on your computer (which you can download [here](https://godotengine.org/download/macos/))
and after doing so, you can import the project from the Godot Hub. Once this is done, you'll be able to launch the project inside of Godot.