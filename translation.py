import time
a=input('enter the english-')
b=input('''choose the language to translate
hi for hindi
fr for french
ru for russian
gu for gujarati
te for telgu
ta for tamil
oriya for oriya
ur for urdu
pa for punjabi
''')

def translate(mm,a):
    from transformers import MarianMTModel, MarianTokenizer
    model_name = ("Helsinki-NLP/opus-mt-en-%s"%mm)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    english_text = a

    inputs = tokenizer(english_text, return_tensors="pt")

    translated_text = model.generate(**inputs)

    decoded_translation = tokenizer.decode(translated_text[0], skip_special_tokens=True)

    print("Eng:", english_text)
    print("%s:"%mm, decoded_translation)
def translate_voice(mm,a):
    from transformers import MarianMTModel, MarianTokenizer
    model_name = ("Helsinki-NLP/opus-mt-en-%s"%mm)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    english_text = a

    inputs = tokenizer(english_text, return_tensors="pt")

    translated_text = model.generate(**inputs)

    decoded_translation = tokenizer.decode(translated_text[0], skip_special_tokens=True)
    from gtts import gTTS
    import os
    time.sleep(1)
    text = decoded_translation
    language = mm
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("output.mp3")
    os.system("start output.mp3")

def summary2(text,lang):
    import nltk
    nltk.download('punkt')
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    input_text = text
    parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
    summarizer = LsaSummarizer()

    summary = summarizer(parser.document, sentences_count=2)

    for sentence in summary:
        translate(lang,str(sentence))

#translate(b,a)
translate_voice(b,a)
#kk=input('enter the text for sumarization')
#mini=int(input('enter the minimum number of words'))
#maxi=int(input('enter the maximum number of words'))
#y=summary(kk,maxi,mini)
#print(y)
#print(summary(y,maxi,len(y)))
#summary2(kk,b)

