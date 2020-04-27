import re
import speech_recognition as speech_recog


sample_audio = speech_recog.AudioFile("audio_data/20200418_200215.wav")

recog = speech_recog.Recognizer()

recognition_speech = open("recognition_speech.txt", "a")

time_offset = 0

while time_offset < 4000:
    text = []

    with sample_audio as audio_file:
        recog.adjust_for_ambient_noise(audio_file)
        audio_content = recog.record(audio_file, duration = 60, offset = time_offset)
    
    text.append(recog.recognize_google(audio_content, language = "ru-RU", show_all = True))

    for word in str(text[0]).split("}"):
        phrase = re.findall(r"transcript': '(.+)'", word)
        
        if len(phrase) != 0:
            recognition_speech.write(phrase[0] + str(time_offset / 60) + "\n")

    time_offset += 60
    