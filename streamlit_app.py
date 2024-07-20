import os
import cv2
import torch
import numpy as np
import streamlit as st
import moviepy.editor as mp
import librosa
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import timm

# Define the ViTLieDetector model class
class ViTLieDetector(nn.Module):
    def __init__(self):
        super(ViTLieDetector, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # Remove the original head
        self.audio_fc = nn.Linear(13 * 400, 768)  # Adjust MFCC input size
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768 * 2, 2)
        )

    def forward(self, x, mfcc):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Merge batch and sequence dimensions
        x = self.vit(x)
        x = x.view(B, T, -1)  # Reshape to (batch_size, sequence_length, num_features)
        x = x.mean(dim=1)  # Average over sequence length

        mfcc = mfcc.view(B, -1)  # Flatten MFCC
        mfcc = self.audio_fc(mfcc)

        combined = torch.cat((x, mfcc), dim=1)  # Concatenate video and audio features
        combined = self.classifier(combined)
        return combined

# Load the saved model
model = ViTLieDetector()
model_path = '/content/drive/MyDrive/final_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Convert MP4 to WAV
def convert_mp4_to_wav(mp4_file, wav_file):
    video = mp.VideoFileClip(mp4_file)
    audio = video.audio
    audio.write_audiofile(wav_file)
    video.close()

# Extract MFCC features from WAV and pad them
def extract_mfcc(wav_file, n_mfcc=13, max_length=400):
    y, sr = librosa.load(wav_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_length:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]
    mfcc = np.expand_dims(mfcc, axis=0)  # Add channel dimension
    return torch.from_numpy(mfcc).float()

# Extract frames from the video
def extract_frames(video_path, sequence_length=12, frame_rate=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    count = 0

    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
        count += 1

    cap.release()
    while len(frames) < sequence_length:
        frames.append(torch.zeros((3, 224, 224)))

    return frames

# Streamlit app
st.title('Lie Detection from Video')
st.write('Upload a video to determine if it contains truth or lie.')

# Upload video file
uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded file
    video_path = os.path.join("uploaded_videos", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert video to WAV
    wav_path = video_path.replace('.mp4', '.wav')
    convert_mp4_to_wav(video_path, wav_path)

    # Extract MFCC features
    mfcc = extract_mfcc(wav_path)
    mfcc = mfcc.to(device).unsqueeze(0)  # Add batch dimension

    # Extract frames
    frames = extract_frames(video_path)
    frames = torch.stack(frames).unsqueeze(0).to(device)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = model(frames, mfcc)
        _, predicted = torch.max(outputs, 1)
        label = "Truth" if predicted.item() == 0 else "Lie"

    st.write(f"The model predicts that the video is: **{label}**")

