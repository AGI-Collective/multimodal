from torchaudio.utils import download_asset
import torchaudio
import io
import torch

#Takes the torchaudio audio
def encode(audio) -> bytes:
    
    stream = io.BytesIO()
    torch.save(audio, stream)
    return stream.getvalue()


def decode(audio):
    
    audio = io.BytesIO(audio)
    data = torch.load(audio)
    return data

    
SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

#it's a tuple of the datapoints and the sampling rate)
data = torchaudio.load(SAMPLE_WAV)
print(data)
encoded = encode(data)
decoded = decode(encoded)
print(decoded)