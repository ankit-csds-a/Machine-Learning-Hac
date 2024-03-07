from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import os

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Translation route
@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        english_text = request.form['english_text']
        target_language = request.form['target_language']

        # Translate text
        translated_text = translate_text(english_text, target_language)

        # Generate voiceover
        if 'generate_voiceover' in request.form:
            generate_voiceover(translated_text, target_language)

        return render_template('result.html', english_text=english_text, target_language=target_language, translated_text=translated_text)

# Translation function
def translate_text(english_text, target_language):
    model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(english_text, return_tensors="pt")
    translated_text = model.generate(**inputs)
    decoded_translation = tokenizer.decode(translated_text[0], skip_special_tokens=True)

    return decoded_translation

# Voiceover generation function
def generate_voiceover(translated_text, target_language):
    tts = gTTS(text=translated_text, lang=target_language, slow=False)
    tts.save("output.mp3")
    os.system("start output.mp3")

if __name__ == '__main__':
    app.run(debug=True)
