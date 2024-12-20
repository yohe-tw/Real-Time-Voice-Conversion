import os
import matplotlib.pyplot as plt
import torchaudio
import torch
import torchaudio.transforms as T



def parse_wav_files(base_dir):
    """
    Parse all .wav files and organize them into a dictionary.

    Args:
        base_dir (str): The base directory containing singers' folders.

    Returns:
        dict: A dictionary with keys as "{song_name + technique}" and values as "{singer, technique, file_path (list)}".
    """
    singers_data = {
        "ZH-Alto-1": "Singer_Alto",
        "ZH-Tenor-1": "Singer_Tenor"
    }

    wav_dict = {}

    for singer_folder, singer_name in singers_data.items():
        singer_path = os.path.join(base_dir, "Chinese", singer_folder)

        # Traverse through techniques
        for technique in os.listdir(singer_path):
            technique_path = os.path.join(singer_path, technique)

            if not os.path.isdir(technique_path):
                continue

            # Traverse through song names
            for song_name in os.listdir(technique_path):
                song_path = os.path.join(technique_path, song_name)

                if not os.path.isdir(song_path):
                    continue

                # Traverse through groups
                for group in os.listdir(song_path):
                    group_path = os.path.join(song_path, group)

                    if not os.path.isdir(group_path):
                        continue

                    # Traverse .wav files in the group
                    for file_name in os.listdir(group_path):
                        if file_name.endswith(".wav"):
                            file_path = os.path.join(group_path, file_name)
                            key = f"{song_name}_{technique}"
                            
                            # Add to dictionary
                            if key not in wav_dict:
                                wav_dict[key] = {
                                    "singer": singer_name,
                                    "technique": technique,
                                    "file_path": []
                                }
                            
                            # Append file path to the list
                            wav_dict[key]["file_path"].append(file_path)

    return wav_dict


def compute_mel_spectrogram(
    audio_path,
    sample_rate=48000,
    n_fft=2048,
    hop_length=512,
    n_mels=256,
    to_db=True,
    show_plot=False
):
    """
    Compute the mel-spectrogram of an audio file.
    
    Parameters:
        audio_path (str): Path to the audio file.
        sample_rate (int): Desired sample rate of the audio.
        n_fft (int): Number of FFT points.
        hop_length (int): Number of samples between successive frames.
        n_mels (int): Number of mel filter banks.
        to_db (bool): Whether to convert the spectrogram to decibel scale.
    
    Returns:
        torch.Tensor: Mel-spectrogram (in dB if to_db is True).
    """
    # Load audio file
    waveform, orig_sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if orig_sample_rate != sample_rate:
        resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)
    
    # Create MelSpectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=35
    )
    
    # Compute the mel-spectrogram
    mel_spec = mel_spectrogram(waveform)
    
    # Convert to dB if required
    if to_db:
        db_transform = torchaudio.transforms.AmplitudeToDB(stype='power')
        mel_spec = db_transform(mel_spec)
        
    # Visualize the mel-spectrogram
    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='dB')
        plt.title('Mel-Spectrogram')
        plt.xlabel('Time (frames)')
        plt.ylabel('Frequency (mel)')
        plt.tight_layout()
        plt.show()
    
    return mel_spec


def compute_global_min_max(data_dict, sample_rate=48000):
    
    global_min = float('inf')
    global_max = float('-inf')
    
    counter = 0

    for key, value in data_dict.items():
        file_paths = value["file_path"]

        for file_path in file_paths:
            
            if counter % 250 == 0:
                print(counter)
            
            # Compute mel-spectrogram for each file
            mel_spectrogram = compute_mel_spectrogram(file_path, sample_rate=sample_rate)

            # Update global min and max
            file_min = mel_spectrogram.min().item()
            file_max = mel_spectrogram.max().item()

            global_min = min(global_min, file_min)
            global_max = max(global_max, file_max)

            # Debugging/logging
            counter = counter + 1
            # print(f"[DEBUG] File: {file_path}, Min: {file_min}, Max: {file_max}")

    return global_min, global_max



def denormalize_mel(normalized_mel, global_min, global_max):
    """
    Reverse the normalization of a mel-spectrogram.

    Args:
        normalized_mel (torch.Tensor): Normalized mel-spectrogram tensor.
        global_min (float): Global minimum value used for normalization.
        global_max (float): Global maximum value used for normalization.

    Returns:
        torch.Tensor: Original mel-spectrogram tensor.
    """
    # Reverse normalization
    original_mel = normalized_mel * (global_max - global_min) + global_min
    return original_mel


def plot_mel_spectrogram_direct(mel_spectrogram, title="Mel-Spectrogram", path="mel-spec.png"):
    """
    Plots the mel-spectrogram.
    
    Args:
        mel_spectrogram (torch.Tensor): The mel-spectrogram to plot.
        title (str): Title of the plot.
    """
    # Handle various possible shapes of mel-spectrogram
    if len(mel_spectrogram.shape) == 4:  # Example: [1, 1, 128, 400]
        mel_spectrogram = mel_spectrogram.squeeze(0).squeeze(0)  # Shape becomes [128, 400]
    elif len(mel_spectrogram.shape) == 3:  # Example: [1, 128, 400]
        mel_spectrogram = mel_spectrogram.squeeze(0)  # Shape becomes [128, 400]
    elif len(mel_spectrogram.shape) == 2:  # Example: [128, 400]
        pass  # Already in the correct shape
    else:
        raise ValueError(f"Invalid mel-spectrogram shape: {mel_spectrogram.shape}. Expected [128, 400].")
    
    # Plot the mel-spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(mel_spectrogram.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.xlabel("Time (Frames)")
    plt.ylabel("Mel Frequency Bins")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.savefig(path)


def stft_loss(predicted, target, n_fft=1024, hop_length=256, win_length=1024):
    """
    Computes the STFT loss between the predicted and target signals.

    Args:
        predicted (torch.Tensor): Predicted mel-spectrogram of shape [batch_size, n_mels, timesteps].
        target (torch.Tensor): Target mel-spectrogram of shape [batch_size, n_mels, timesteps].
        n_fft (int): Number of FFT components.
        hop_length (int): Hop length for STFT.
        win_length (int): Window length for STFT.

    Returns:
        torch.Tensor: The STFT loss.
    """
    # Ensure the input has the correct shape
    if len(predicted.shape) != 3 or len(target.shape) != 3:
        raise ValueError(f"Invalid input shapes: predicted={predicted.shape}, target={target.shape}. Expected [batch_size, n_mels, timesteps].")

    # Compute STFT for predicted and target signals
    predicted_stft = torch.stft(predicted.flatten(start_dim=1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=False)
    target_stft = torch.stft(target.flatten(start_dim=1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=False)

    # Compute magnitude
    predicted_mag = torch.sqrt(predicted_stft.pow(2).sum(-1))
    target_mag = torch.sqrt(target_stft.pow(2).sum(-1))

    # Compute STFT loss
    loss = torch.nn.functional.mse_loss(predicted_mag, target_mag)

    return loss


from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as bundle

def convert_mel_to_wav(mel_spectrogram, output_path):
    """
    Convert a mel-spectrogram to a .wav file using torchaudio's HiFiGAN vocoder.

    Args:
        mel_spectrogram (torch.Tensor): Input mel-spectrogram with shape [n_mels, timesteps].
        output_path (str): File path to save the generated .wav file.
    """
    import matplotlib.pyplot as plt

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained HiFi-GAN vocoder
    vocoder = bundle.get_vocoder().to(device)
    vocoder.eval()

    print(f"mel_spectrogram.shape: {mel_spectrogram.shape}")

    with torch.no_grad():
        # Ensure mel-spectrogram is of the correct shape [batch_size, n_mels, timesteps]
        if len(mel_spectrogram.shape) == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)  # Add batch dimension

        # Move the mel-spectrogram to the same device as the model
        mel_spectrogram = mel_spectrogram.to(device)

        # Ensure the number of mel bins matches the expected value
        if mel_spectrogram.size(1) != bundle._vocoder_params["in_channels"]:
            raise ValueError(f"Expected {bundle._vocoder_params['in_channels']} mel bins, but got {mel_spectrogram.size(1)}.")

        # Pass the mel-spectrogram through the vocoder
        waveform = vocoder(mel_spectrogram)
        print(f"waveform.shape: {waveform.shape}")

        # Reshape waveform for plotting and saving
        waveform = waveform.squeeze().cpu()  # Remove all singleton dimensions
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension for mono audio

        # Plot the waveform
        plt.figure(figsize=(10, 4))
        for i, channel in enumerate(waveform):
            plt.plot(channel.numpy(), label=f"Channel {i + 1}")
        plt.title("Waveform")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

        # Save the waveform
        torchaudio.save(output_path, waveform, bundle.sample_rate)

    print(f"Generated audio saved to: {output_path}")

