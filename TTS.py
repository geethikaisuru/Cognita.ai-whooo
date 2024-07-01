from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import numpy as np
import librosa
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

text = extract_text_from_pdf(pdf_path="inputs/cognita_test_lite.pdf")
print("Text extraction completed.")

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text=text, return_tensors="pt")

def extract_voice_characteristics(audio_files, output_file, sr=22050, embedding_dim=512):
    """
    Extracts speaker's voice characteristics from multiple audio files and saves
    the averaged embedding as a .npy file.

    Parameters:
    audio_files (list): List of paths to the audio files.
    output_file (str): Path to save the .npy file.
    sr (int): Sample rate for loading the audio file. Default is 22050.
    embedding_dim (int): Dimensionality of the embedding to extract. Default is 512.
    """
    embeddings = []
    
    for audio_file in audio_files:
        y, sr = librosa.load(audio_file, sr=sr)
        embedding = extract_embedding(y, sr, embedding_dim)
        embeddings.append(embedding)
    
    # Average the embeddings
    averaged_embedding = np.mean(embeddings, axis=0)
    
    # Convert the embedding to a torch tensor of size [1, embedding_dim]
    embedding_tensor = torch.tensor(averaged_embedding).unsqueeze(0)
    
    # Save the embedding tensor to a .npy file
    np.save(output_file, embedding_tensor.numpy())

def extract_embedding(y, sr, embedding_dim):
    """
    Extracts a 512-dimensional embedding from audio data.

    Parameters:
    y (np.ndarray): Audio signal.
    sr (int): Sample rate.
    embedding_dim (int): Dimensionality of the embedding to extract.

    Returns:
    np.ndarray: A 512-dimensional embedding.
    """
    n_mfcc = 13  # Example number of MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    embedding = mfccs.flatten()[:embedding_dim]  # Example flattening to 512 dimensions

    return embedding

# List of audio files
audio_files = ['inputs/New/Rec1.wav', 'inputs/New/Rec2.wav', 'inputs/New/Rec3.wav']
output_file = 'Voice_npy/AvgVC.npy'
extract_voice_characteristics(audio_files, output_file)



local_embeddings = np.load('Voice_npy/AvgVC.npy')
speaker_embeddings = torch.tensor(local_embeddings)
print(speaker_embeddings.shape)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embedding(embedding_file, method='tsne'):
  # Load the embedding
    embedding = np.load(embedding_file)
    
    if method == 'tsne':
        # Use t-SNE to reduce the embedding to 2D
        tsne = TSNE(n_components=2, random_state=42)
        embedding_2d = tsne.fit_transform(embedding)
    elif method == 'pca':
        # Use PCA to reduce the embedding to 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(embedding)
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'pca'.")

    # Plot the 2D embedding
    plt.figure(figsize=(8, 8))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c='blue', marker='o')
    plt.title(f'{method.upper()} Visualization of Voice Embedding')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# Visualize the embedding
visualize_embedding('Voice_npy/AvgVC.npy', method='tsne')


speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
sf.write("outputs/AvgVoice.wav", speech.numpy(), samplerate=16000)