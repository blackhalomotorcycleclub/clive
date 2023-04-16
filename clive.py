import nltk
from nltk.chat.util import Chat, reflections
from irc.bot import SingleServerIRCBot
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class DeepLearningChatbot(SingleServerIRCBot):
    def __init__(self, server, channel, nickname):
        super().__init__([(server, 6667)], nickname, nickname)
        self.channel = channel
        self.tokenizer = Tokenizer()
        self.max_len = 20
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.max_len, 1)))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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

    def respond(self, message):
        message = self._preprocess([message])
        prediction = self.model.predict(message)
        return 'yes' if prediction[0][0] > 0.5 else 'no'

    def _preprocess(self, messages):
        self.tokenizer.fit_on_texts(messages)
        sequences = self.tokenizer.texts_to_sequences(messages)
        return pad_sequences(sequences, maxlen=self.max_len)

def setup():
    server = input('Enter IRC server: ')
    channel = input('Enter IRC channel: ')
    nickname = input('Enter chatbot nickname: ')
    chatbot = DeepLearningChatbot(server, channel, nickname)
    chatbot.start()

setup()