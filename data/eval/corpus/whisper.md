# Whisper Speech Recognition

Whisper is an automatic speech recognition (ASR) system released by OpenAI in 2022. It is an encoder-decoder transformer trained on 680,000 hours of multilingual and multitask supervised data collected from the web. Audio is resampled to 16 kHz, converted to an 80-channel log-Mel spectrogram and processed in 30-second windows.

Whisper is multitask: the same model performs transcription, translation into English, language identification and voice activity detection, controlled by special tokens in the decoder prompt. Model sizes range from tiny (39 million parameters) to large (1.55 billion parameters); larger models are more accurate but slower.

Because the encoder sees fixed 30-second windows, longer recordings are transcribed by chunking the audio, optionally with overlapping strides so words at the chunk boundary are not lost. Whisper predicts timestamp tokens, which allows segment-level start and end times to be recovered from the decoded output.

Common failure modes include hallucinated text on silence or music, repeated phrases in long-form transcription, and lower accuracy on low-resource languages and heavy accents.
