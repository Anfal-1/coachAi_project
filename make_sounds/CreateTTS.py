from gtts import gTTS

tts = gTTS(text="down", lang="en")
tts.save("../resources/sounds/down.mp3")
