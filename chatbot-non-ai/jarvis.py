import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import os
import smtplib
from random import randrange
engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
#print(voices[0].id)
engine.setProperty('voice',voices[0].id)#0 for male 1 for female
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
def wishMe():
    hour=int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning Sir!")
    elif hour>=12 and hour<18:
        speak("Good Afternoon Sir!")
    else:
        speak("Good Evening Sir!")
        print("Jarvis Online Ready for Your Command\nHow May I help you?")
    speak("Jarvis Online Ready for Your Command")
    speak("How May I help you?")
def sendEmail(to,content):
    server=smtplib.SMTP('smtp.gmail.com',587)
    server.ehlo()
    server.starttls()
    server.login('youremail@gmail.com-here','your-password-here')
    server.sendmail('youremail@gmail.com',to,content)
    server.close()

def takeCommand():
    #it takes microphone input from user and returns string output
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold=1 # seconds of non-speaking audio before a phrase is considered complete
        audio=r.listen(source)
    try:
        print("Recognizing...")
        query=r.recognize_google(audio)#,language='en-in')
        print(f"User said: {query}\n")
    except Exception as e:
        print(e)
        print("Say That Again please...")
        return "None"
    return query

if __name__ == '__main__':
    wishMe()
    run=True
    while(run):
        query=takeCommand().lower()
    #logic for executing taska based on query

        if 'wikipedia' in query:
            speak("Searching Wikipedia...")
            print("Searching Wikipedia...")
            query=query.replace("wikipedia","")
            results=wikipedia.summary(query,sentences=2)
            speak("According to Wikipedia")
            print("According to wikipedia")
            print(results)
            speak(results)
        elif 'open youtube' in query:
            webbrowser.open("youtube.com")
        elif 'open google' in query:
            webbrowser.open("google.com")
        elif 'play music' in query:
            music_dir="D:\\Music"
            songs=os.listdir(music_dir)
            os.startfile(os.path.join(music_dir,songs[randrange(0,len(songs)-1)]))
        elif 'the time' in query:
            strTime=datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"Sir The time is {strTime}")
        elif 'send email to rudy' in query:
            try:
                speak("What should I say?")
                content=takeCommand()
                to="rchoudhuri500@gmail.com"
                sendEmail(to,content)
                speak("Email has been sent")
            except Exception as e:
                speak("Not able to send email at the moment")
        elif 'quit' in query:
            speak("Hope I was of your use ,Thank you Sir,hope to See you again soon!!!")
            run=False





