# Simple NLP Chatbot

This is a simple NLP chatbot that classifies user input into different categories such as greetings, goodbyes, and thanks. The chatbot uses pre-trained word embeddings from the Google News dataset and a basic neural network to make predictions.

## Dependencies

- Python 3.7 or higher
- TensorFlow 2.0 or higher
- Gensim
- NumPy

Run in the terminal to install dependencies from the main project folder.

```commandline
pip install -r requirements.txt
```

## Usage

1. Save the chatbot code in a file named `chatbot.py`.
2. Download the pre-trained Google News word vectors by clicking [here](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz).
3. Once the download is complete, locate the downloaded `GoogleNews-vectors-negative300.bin.gz` file in your Downloads folder (or the folder you chose for the download).
4. Move the `GoogleNews-vectors-negative300.bin.gz` file to the directory containing your `chatbot.py` file.
5. Run the file with:
```python
python chatbot.py
```

6. The chatbot will start and prompt you for input. Type your message and press Enter to see the chatbot's response. To exit the chatbot, type 'exit' and press Enter.

## Customization

To customize the chatbot, you can modify the dataset and categories in the `chatbot.py` file. Add more examples to the dataset and update the categories as needed to improve the chatbot's performance.

Remember that this model is quite basic, and its performance might not be optimal. Expanding the dataset and improving the neural network architecture will further enhance the model's performance.
