import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

url_confidence = #
text_confidence = #

url_input = Input(shape=(1,), name='url_input')
text_input = Input(shape=(1,), name='text_input')

url_hidden = Dense(8, activation='relu')(url_input)
text_hidden = Dense(8, activation='relu')(text_input)

combined = concatenate([url_hidden, text_hidden])

output = Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[url_input, text_input], outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit({'url_input': x_train_url, 'text_input': x_train_text}, y_train, epochs=10, batch_size=32)

url_test_confidence = 0.9
text_test_confidence = 0.8
prediction = model.predict({'url_input': [url_test_confidence], 'text_input': [text_test_confidence]})
