
# *************************************************
# try python stuff 
# *************************************************
import cmu_sphinx4

audio_URL   = 'audio.wav'
transcriber = cmp_sphinx4.Transcriber(autdio_URL)
for line in transcriber.transcript_stream():
        print line
 
 
