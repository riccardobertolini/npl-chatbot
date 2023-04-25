import random
import tensorflow as tf
import numpy as np
import gensim

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

def sentence_to_vectors(sentence):
    words = sentence.split()
    vectors = []
    for word in words:
        try:
            vectors.append(word2vec_model[word])
        except KeyError:
            pass
    if not vectors:
        return [0.0] * 300
    return np.mean(vectors, axis=0)

categories = {
    'greeting': 0,
    'goodbye': 1,
    'thanks': 2,
    'weather': 3,
    'default': 4,
}

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(300,), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(categories), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train = [sentence_to_vectors(d[0]) for d in data]
y_train = tf.keras.utils.to_categorical([categories[d[1]] for d in data])

model.fit(np.array(X_train), y_train, epochs=50)

def predict_category(sentence):
    vector = sentence_to_vectors(sentence)
    prediction = model.predict(np.array([vector]))[0]
    category_index = np.argmax(prediction)
    return list(categories.keys())[list(categories.values()).index(category_index)]

def nlp_chat():
    print('Hello! How can I help you today?')

    while True:
        user_input = input('> ').lower()
        user_input = re.sub(r'[^\w\s]', '', user_input)

        if 'exit' in user_input:
            print('Goodbye!')
            break

        category = predict_category(user_input)

        if category == 'greeting':
            print(random.choice(['Hello!', 'Hi there!', 'Hi!', 'Hey!']))
        elif category == 'goodbye':
            print(random.choice(['Goodbye!', 'Bye!', 'See you later!', 'Adios!']))
        elif category == 'thanks':
            print(random.choice(['You are welcome!', 'No problem!', 'My pleasure!']))
        elif category == 'weather':
            print("I'm sorry, I cannot provide real-time weather information.")
        else:
            print(random.choice(['I am sorry, I did not understand what you said.',
                                 'Could you please rephrase that?',
                                 'I am not sure what you are asking.']))

if __name__ == '__main__':
    nlp_chat()
