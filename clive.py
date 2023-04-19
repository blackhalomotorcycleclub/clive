import nltk
from nltk.chat.util import Chat, reflections
from irc.bot import SingleServerIRCBot
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

class DeepLearningChatbot(SingleServerIRCBot):
    def __init__(self, server, channel, nickname):
        super().__init__([(server, 6667)], nickname, nickname)
        self.channel = channel
        self.nickname = nickname
        self.tokenizer = Tokenizer()
        self.max_len = 20
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.max_len, 1)))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.load_brain()

    def on_welcome(self, connection, event):
        connection.join(self.channel)

    def on_pubmsg(self, connection, event):
        message = event.arguments[0]
        response = self.respond(message)
        if response:
            connection.privmsg(self.channel, response)

    def train(self, inputs, outputs, epochs=10):
        inputs = self._preprocess(inputs)
        outputs = np.array(outputs)
        self.model.fit(inputs, outputs, epochs=epochs)
        self.save_brain()

    def respond(self, message):
        message = self._preprocess([message])
        prediction = self.model.predict(message)
        return 'yes' if prediction[0][0] > 0.5 else 'no'

    def _preprocess(self, messages):
        self.tokenizer.fit_on_texts(messages)
        sequences = self.tokenizer.texts_to_sequences(messages)
        return pad_sequences(sequences, maxlen=self.max_len)

    def save_brain(self):
        filename = f"{self.nickname}.brain"
        with open(filename, 'wb') as f:
            pickle.dump((self.tokenizer.word_index, self.model.get_weights()), f)

    def load_brain(self):
        filename = f"{self.nickname}.brain"
        try:
            with open(filename, 'rb') as f:
                word_index, weights = pickle.load(f)
            self.tokenizer.word_index = word_index
            self.model.set_weights(weights)
        except FileNotFoundError:
            pass

def collect_training_data(server, channel, nickname):
    inputs = []
    outputs = []
    bot = SingleServerIRCBot([(server, 6667)], nickname, nickname)
    def on_pubmsg(connection,event):
        message = event.arguments[0]
        inputs.append(message)
        response = input(f"Enter response for '{message}': ")
        outputs.append(response)
    bot.on_pubmsg = on_pubmsg
    bot.start()
    return inputs, outputs

def setup():
    server = input('Enter IRC server: ')
    channel = input('Enter IRC channel: ')
    nickname = input('Enter chatbot nickname: ')
    chatbot = DeepLearningChatbot(server, channel,nickname)
    chatbot.start()

setup()
