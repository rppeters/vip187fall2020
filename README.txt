A quick explanation of the purpose of each sound file:

spectrogramProcessing.py:

(Attempts to) Extract vowels from speech, apply vertical edge detection for spectrogram enhancement, and finally hysterisis for edge chaining to approximate formants.
Currently in the process of designing an interpolation method to extract formant frequencies from the low resolution edges.

SpectrogramPractice.py 

Basic python program to display the spectrogram of a sound file

matchFilter.py

A first attempt at implementing the match filter used in spectrogramProcessing.py for vowel extraction. The function implemented for match filter can be used for other templates as well

impulseDetection.py

An attempt at using edge detection on the wave form to detect impulse waves (claps, etc). I ended up breaking it when I accidentally used a vowel as the input and realized I could calculate
the fundamental frequency (F0) from the edge detection results, but not its not functional.
