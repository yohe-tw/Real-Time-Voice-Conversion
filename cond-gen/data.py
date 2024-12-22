# total 4080
from torch.utils.data import Dataset, DataLoader, random_split
import os
import random
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize

class VocalTechniqueDataset(Dataset):
    def __init__(self, data_dict, sample_rate=48000, fixed_timesteps=196, step_size=10, 
                 global_mel_min=-100, global_mel_max=54, transform=None, num_duplicates=5):
        
        """
        Initializes the dataset.

        Args:
            data_dict (dict): Dictionary containing file paths for all techniques.
            sample_rate (int): Sample rate for loading audio.
            fixed_timesteps (int): Fixed number of timesteps for all samples.
            step_size (int): Step size for sequential sampling.
            transform (callable, optional): Transformation to apply to audio.
        """
        
        print("Initializing VocalTechniqueDataset")
        
        self.data_pairs = []
        self.sample_rate = sample_rate
        self.fixed_timesteps = fixed_timesteps
        self.step_size = step_size
        self.transform = transform
        self.global_mel_min = global_mel_min
        self.global_mel_max = global_mel_max

        # techniques list
        techniques = ["Breathy", "Glissando", "Mixed_Voice_and_Falsetto", "Pharyngeal", "Vibrato"]
        self.technique_to_onehot = {tech: i for i, tech in enumerate(techniques)}

        for key, value in data_dict.items():
            # Extract control and technique files
            control_files = [path for path in value["file_path"] if "Control_Group" in path]
            technique_files = [path for path in value["file_path"] if "Group" in path and "Control_Group" not in path]

            control_files_map = {os.path.basename(f): f for f in control_files}

            for technique_path in technique_files:
                filename = os.path.basename(technique_path)
                if filename in control_files_map:
                    control_path = control_files_map[filename]

                    # Extract the technique folder name of the target
                    target_technique_folder = os.path.basename(os.path.dirname(technique_path))  # e.g., Falsetto_Group

                    # Find references from other songs but the same technique
                    available_references = []
                    for other_key, other_value in data_dict.items():
                        if other_key != key:  # Ensure different songs
                            other_technique_files = [
                                path for path in other_value["file_path"]
                                if target_technique_folder in path  # Match exact technique folder
                            ]
                            available_references.extend(other_technique_files)

                    # Ensure enough references exist
                    if len(available_references) < num_duplicates:
                        raise ValueError(f"Not enough references for {technique_path} with technique {target_technique_folder}.")

                    # Select references
                    selected_references = random.sample(available_references, num_duplicates)
                    for reference_path in selected_references:
                        pair = (control_path, reference_path, technique_path, value["technique"])
                        self.data_pairs.append(pair)

                        # Debug print for clarity
                        # print(f"Added pair: {pair}")
                        
            
    def __len__(self):
        return len(self.data_pairs)

    def sample_fixed_window(self, control_mel, reference_mel, technique_mel, target_timesteps = 256):
        """
        Randomly selects a window of 400 timesteps from all three mel-spectrograms.

        Args:
            control_mel (torch.Tensor): Mel-spectrogram of shape [n_mels, timesteps].
            reference_mel (torch.Tensor): Mel-spectrogram of shape [n_mels, timesteps].
            technique_mel (torch.Tensor): Mel-spectrogram of shape [n_mels, timesteps].

        Returns:
            torch.Tensor: Adjusted control mel-spectrogram of shape [n_mels, 400].
            torch.Tensor: Adjusted reference mel-spectrogram of shape [n_mels, 400].
            torch.Tensor: Adjusted technique mel-spectrogram of shape [n_mels, 400].
        """
        _, control_timesteps = control_mel.shape
        _, reference_timesteps = reference_mel.shape
        _, technique_timesteps = technique_mel.shape

        # Ensure all mel-spectrograms have enough timesteps
        min_timesteps = min(control_timesteps, reference_timesteps, technique_timesteps)
        if min_timesteps < target_timesteps:
            raise ValueError("All mel-spectrograms must have at least 400 timesteps.")

        # Randomly select a starting index for the window
        start_idx = random.randint(0, min_timesteps - target_timesteps)
        end_idx = start_idx + target_timesteps

        # Slice the window from each mel-spectrogram
        control_mel_window = control_mel[:, start_idx:end_idx]
        reference_mel_window = reference_mel[:, start_idx:end_idx]
        technique_mel_window = technique_mel[:, start_idx:end_idx]

        return control_mel_window, reference_mel_window, technique_mel_window
    
    def pitch_match(self, control_waveform, reference_waveform, target_waveform, sample_rate=48000):
        
        
        """
        Align the pitch of reference and target waveforms to match the control waveform.

        Args:
            control_waveform (torch.Tensor): Control waveform tensor [1, time].
            reference_waveform (torch.Tensor): Reference waveform tensor [1, time].
            target_waveform (torch.Tensor): Target waveform tensor [1, time].
            sample_rate (int): Sampling rate of the audio.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pitch-aligned reference and target waveforms.
        """
        
        print("in pitch_match")
        
        # Ensure mono audio for simplicity
        def to_mono(waveform):
            if waveform.dim() > 1 and waveform.shape[0] > 1:
                return torch.mean(waveform, dim=0, keepdim=True)
            return waveform

        control_waveform = to_mono(control_waveform)
        reference_waveform = to_mono(reference_waveform)
        target_waveform = to_mono(target_waveform)

        # Detect fundamental frequency (F0) for each waveform
        def detect_f0(waveform, sample_rate):
            pitch = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate, frame_time=0.032)
            valid_pitch = pitch[pitch > 0]  # Ignore unvoiced frames
            if len(valid_pitch) > 0:
                return torch.median(valid_pitch).item()  # Median pitch
            else:
                return 0  # No pitch detected

        control_f0 = detect_f0(control_waveform, sample_rate)
        reference_f0 = detect_f0(reference_waveform, sample_rate)
        target_f0 = detect_f0(target_waveform, sample_rate)

        print(f"Control Pitch (F0): {control_f0:.2f} Hz")
        print(f"Reference Pitch (F0): {reference_f0:.2f} Hz")
        print(f"Target Pitch (F0): {target_f0:.2f} Hz")

        # Helper function to pitch shift waveform
        def pitch_shift_to_f0(waveform, source_f0, target_f0, sample_rate):
            if source_f0 > 0 and target_f0 > 0:
                pitch_shift_semitones = 12 * torch.log2(torch.tensor(target_f0 / source_f0)).item()
                print(f"Pitch Shift: {pitch_shift_semitones:.2f} semitones")
                return torchaudio.functional.pitch_shift(waveform, sample_rate=sample_rate, n_steps=pitch_shift_semitones)
            else:
                print("Pitch could not be detected. Returning original waveform.")
                return waveform

        # Align reference and target waveforms to control's pitch
        aligned_reference_waveform = pitch_shift_to_f0(reference_waveform, reference_f0, control_f0, sample_rate)
        aligned_target_waveform = pitch_shift_to_f0(target_waveform, target_f0, control_f0, sample_rate)

        return control_waveform, aligned_reference_waveform, aligned_target_waveform
    
    def pitch_shift(self, control_waveform, reference_waveform, target_waveform, sample_rate=48000):
        """
        Apply the same pitch shift to control, reference, and target waveforms.

        Args:
            control_waveform (torch.Tensor): Control waveform tensor.
            reference_waveform (torch.Tensor): Reference waveform tensor.
            target_waveform (torch.Tensor): Target waveform tensor.
            sample_rate (int): Sampling rate for the audio.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Pitch-shifted control, reference, and target waveforms.
        """
        # Decide the pitch shift in semitones
        if random.random() < 0.5:
            n_steps = random.randint(-6, -1)
        else:
            n_steps = random.randint(1, 6)

        print(f"Applying same pitch shift: {n_steps} semitones")

        # Create the pitch shifter
        pitch_shifter = torchaudio.transforms.PitchShift(sample_rate=sample_rate, n_steps=n_steps)

        # Apply pitch shift to all waveforms
        control_waveform = pitch_shifter(control_waveform)
        reference_waveform = pitch_shifter(reference_waveform)
        target_waveform = pitch_shifter(target_waveform)

        return control_waveform, reference_waveform, target_waveform

    def time_stretch_with_pitch_correction(self, control_waveform, reference_waveform, target_waveform, sample_rate=48000):
        
        """
        Apply the same time stretch and pitch correction to control and target waveforms.

        Args:
            control_waveform (torch.Tensor): Control waveform tensor.
            target_waveform (torch.Tensor): Target waveform tensor.
            sample_rate (int): Sampling rate for the audio.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Time-stretched and pitch-corrected control and target waveforms.
        """
        
        # Decide the time stretch rate
        if random.random() < 0.5:
            rate = random.choice([0.7, 0.75, 0.8, 0.85, 0.9])
        else:
            rate = random.choice([1.1, 1.15, 1.2, 1.25, 1.3])

        print(f"Applying same time stretch with pitch correction: {rate} faster/slower")

        # Function to apply time stretch and pitch correction
        def stretch_and_correct(waveform, rate, sample_rate):
            waveform_np = waveform.squeeze().numpy()
            original_length = len(waveform_np)
            new_length = int(original_length / rate)
            stretched_waveform_np = resize(waveform_np, (new_length,), preserve_range=True)

            stretched_waveform = torch.tensor(stretched_waveform_np, dtype=torch.float32).unsqueeze(0)
            semitones = -12 * torch.log2(torch.tensor(rate))
            pitch_shift_transform = torchaudio.transforms.PitchShift(sample_rate=sample_rate, n_steps=semitones.item())

            return pitch_shift_transform(stretched_waveform)

        # Apply to both control and target waveforms
        control_waveform = stretch_and_correct(control_waveform, rate, sample_rate)
        target_waveform = stretch_and_correct(target_waveform, rate, sample_rate)

        return control_waveform, reference_waveform, target_waveform

    def add_noise(self, control_waveform, reference_waveform, target_waveform, sample_rate=48000):
        """
        Apply Gaussian noise to control and reference waveforms, leaving the target waveform unmodified.

        Args:
            control_waveform (torch.Tensor): Control waveform tensor.
            reference_waveform (torch.Tensor): Reference waveform tensor.
            target_waveform (torch.Tensor): Target waveform tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Control and reference waveforms with noise added,
                                                             target waveform unmodified.
        """
        noise_level = 0.01

        # Apply noise to control waveform
        control_noise = torch.randn_like(control_waveform) * noise_level * control_waveform.abs().max()
        control_waveform = control_waveform + control_noise
        print(f"Added Gaussian noise to control with level {noise_level}")

        # Apply noise to reference waveform
        reference_noise = torch.randn_like(reference_waveform) * noise_level * reference_waveform.abs().max()
        reference_waveform = reference_waveform + reference_noise
        print(f"Added Gaussian noise to reference with level {noise_level}")

        # Target waveform remains unmodified
        return control_waveform, reference_waveform, target_waveform

    def time_mask(self, control_waveform, reference_waveform, target_waveform, sample_rate=48000):
        """
        Apply random time masks independently to control, reference, and target waveforms.

        Args:
            control_waveform (torch.Tensor): Control waveform tensor.
            reference_waveform (torch.Tensor): Reference waveform tensor.
            target_waveform (torch.Tensor): Target waveform tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Control, reference, and target waveforms with time masks applied.
        """
        def mask_waveform(waveform):
            n_mels, time_steps = waveform.shape
            num_masks = random.randint(5, 15)

            for _ in range(num_masks):
                mask_width = random.choice(range(5, 20))
                mask_start = random.randint(0, time_steps - mask_width)
                print(f"Masking waveform: from {mask_start} to {mask_start + mask_width}")
                waveform[:, mask_start:mask_start + mask_width] = -100.0

            return waveform

        # Independently mask each waveform
        control_waveform = mask_waveform(control_waveform)
        reference_waveform = mask_waveform(reference_waveform)
        target_waveform = mask_waveform(target_waveform)

        return control_waveform, reference_waveform, target_waveform

    def frequency_filter(self, control_waveform, reference_waveform, target_waveform, sample_rate=48000):
        """
        Apply a frequency filter to control and reference waveforms, leaving the target waveform unmodified.

        Args:
            control_waveform (torch.Tensor): Control waveform tensor.
            reference_waveform (torch.Tensor): Reference waveform tensor.
            target_waveform (torch.Tensor): Target waveform tensor.
            sample_rate (int): Sampling rate for the audio.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Control and reference waveforms with frequency filtering applied,
                                                             target waveform unmodified.
        """
        low_freq = 200
        high_freq = 2000
        central_freq = (low_freq + high_freq) / 2
        Q = 1.0

        # Apply frequency filter to control waveform
        print("Applying frequency filter to control waveform")
        control_waveform = torchaudio.functional.bandpass_biquad(control_waveform, sample_rate, central_freq, Q)

        # Apply frequency filter to reference waveform
        print("Applying frequency filter to reference waveform")
        reference_waveform = torchaudio.functional.bandpass_biquad(reference_waveform, sample_rate, central_freq, Q)

        # Target waveform remains unmodified
        return control_waveform, reference_waveform, target_waveform
    
    # not use
    """
    def transform_to_mel_and_mask_bins(self, waveform, sample_rate, n_mels=256, augment_prob=0.5, 
                                       f_min=72, f_max=None, hop_length=512, n_fft=2048):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension

        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, 
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            hop_length=hop_length, 
            n_fft=n_fft
        )
        mel_spectrogram = mel_transform(waveform).squeeze(0)  # Remove batch dimension
        print(f"Mel-spectrogram shape: {mel_spectrogram.shape}")

        if augment_prob < 0.5:
            print("Applying augmentation: masking random mel bins")
            num_bins_to_mask = random.randint(10, 20)
            for _ in range(num_bins_to_mask):
                mel_bin = random.randint(0, n_mels - 1)
                mel_spectrogram[mel_bin, :] = 0.0
        else:
            print("No augmentation applied")

        return mel_spectrogram
    """
    
    
    def augment_data(self, control_waveform, reference_waveform, target_waveform, sample_rate=48000):
        
        """
        Apply random augmentation to the control, reference, and target audio.

        Args:
            control_path (str): Path to the control audio.
            reference_path (str): Path to the reference audio.
            target_path (str): Path to the target audio.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Augmented control, reference, and target waveforms.
        """
        
        print("in augment_data")
        
        if random.random() < 0.5:
            print(f"Applying augmentation: time_stretch")
            control_waveform, reference_waveform, target_waveform = \
                self.time_stretch_with_pitch_correction(control_waveform, reference_waveform, target_waveform, sample_rate=48000)

        
        if random.random() < 0.5:
            print(f"Applying augmentation: pitch_shift")
            control_waveform, reference_waveform, target_waveform = \
                self.pitch_shift(control_waveform, reference_waveform, target_waveform, sample_rate=48000)
            
        if random.random() < 0.5:
            print(f"Applying augmentation: frequency_filter")
            control_waveform, reference_waveform, target_waveform = \
                self.frequency_filter(control_waveform, reference_waveform, target_waveform, sample_rate=48000)
            
        if random.random() < 0.5:
            print(f"Applying augmentation: time_mask")
            control_waveform, reference_waveform, target_waveform = \
                self.time_mask(control_waveform, reference_waveform, target_waveform, sample_rate=48000)
            
        if random.random() < 0.5:
            print(f"Applying augmentation: add_noise")
            control_waveform, reference_waveform, target_waveform = \
                self.add_noise(control_waveform, reference_waveform, target_waveform, sample_rate=48000)
            
        """
        if random.random() < 0.5:
            print(f"Applying augmentation: transform_to_mel_and_mask_bins to {name}")
            waveform = transform_to_mel_and_mask_bins(waveform, self.sample_rate)
        """
            

        return control_waveform, reference_waveform, target_waveform
    
    def compute_mel_spectrogram(
        self,
        waveform,
        sample_rate=48000,
        n_fft=2048,
        hop_length=512,
        n_mels=256,
        to_db=True,
        show_plot=False
    ):
        """
        Compute the mel-spectrogram of a waveform tensor.

        Parameters:
            waveform (torch.Tensor): Input waveform tensor of shape [channels, samples].
            sample_rate (int): Desired sample rate of the audio.
            n_fft (int): Number of FFT points.
            hop_length (int): Number of samples between successive frames.
            n_mels (int): Number of mel filter banks.
            to_db (bool): Whether to convert the spectrogram to decibel scale.
            show_plot (bool): Whether to display the mel-spectrogram plot.

        Returns:
            torch.Tensor: Mel-spectrogram (in dB if to_db is True).
        """
        # Ensure waveform is in correct shape: [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension if missing

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
    
    def normalize_mel(self, mel):
        """
        Normalize the mel-spectrogram to range [0, 1].
        Args:
            mel (np.ndarray or torch.Tensor): Input mel-spectrogram.
        Returns:
            torch.Tensor: Normalized mel-spectrogram.
        """
        return (mel - self.global_mel_min) / (self.global_mel_max - self.global_mel_min + 1e-8)
    
    
    def __getitem__(self, idx):
        
        control_path, reference_path, target_path, technique = self.data_pairs[idx]
        # print(f"control_path: {control_path}")
        # print(f"reference_path: {reference_path}")
        # print(f"target_path: {target_path}")
        
        # Load waveforms
        control_waveform, sample_rate = torchaudio.load(control_path)
        reference_waveform, sample_rate = torchaudio.load(reference_path)
        target_waveform, sample_rate = torchaudio.load(target_path)
        
        # match pitch
        # control_waveform, reference_waveform, target_waveform = self.pitch_match(control_waveform, reference_waveform, target_waveform)
        
        # data augmentation
        # control_waveform, reference_waveform, target_waveform = self.augment_data(control_waveform, reference_waveform, target_waveform)
        
        # Load audio
        control_mel = self.compute_mel_spectrogram(control_waveform, sample_rate=self.sample_rate)
        reference_mel = self.compute_mel_spectrogram(reference_waveform, sample_rate=self.sample_rate)
        technique_mel = self.compute_mel_spectrogram(target_waveform, sample_rate=self.sample_rate)
        
        # print("--------------- before shape operations 1 ---------------")
        # print(f"control_mel.shape: {control_mel.shape}")
        # print(f"reference_mel.shape: {reference_mel.shape}")
        # print(f"technique_mel.shape: {technique_mel.shape}")
        
        # print("--------------- after shape operations 1 ---------------")
        # Remove the extra batch dimension if present
        control_mel = control_mel.squeeze(0)  # Shape becomes [n_mels, timesteps]
        reference_mel = reference_mel.squeeze(0)
        technique_mel = technique_mel.squeeze(0)  # Shape becomes [n_mels, timesteps]
        # print(f"control_mel.shape: {control_mel.shape}")
        # print(f"reference_mel.shape: {reference_mel.shape}")
        # print(f"technique_mel.shape: {technique_mel.shape}")
        
        # Normalize mel-spectrograms
        control_mel = self.normalize_mel(control_mel)
        reference_mel = self.normalize_mel(reference_mel)
        technique_mel = self.normalize_mel(technique_mel)
        
        # print("--------------- before shape operations 2 ---------------")
        # print(f"control_mel.shape: {control_mel.shape}")
        # print(f"reference_mel.shape: {reference_mel.shape}")
        # print(f"technique_mel.shape: {technique_mel.shape}")
        
        # get windowed mel.
        control_mel, reference_mel, technique_mel = self.sample_fixed_window(control_mel, reference_mel, technique_mel)
        
        # Ensure correct shape: [1, 256, 400] → Add batch and channel dimension
        control_mel = control_mel.unsqueeze(0)  # Shape: [1, 256, fixed_timesteps] → [1, 256, 400]
        reference_mel = reference_mel.unsqueeze(0)
        technique_mel = technique_mel.unsqueeze(0)
        # print("--------------- after shape operations 2 ---------------")
        # print(f"control_mel.shape: {control_mel.shape}")
        # print(f"reference_mel.shape: {reference_mel.shape}")
        # rint(f"technique_mel.shape: {technique_mel.shape}")

        technique_onehot = self.technique_to_onehot[technique]
        # print(technique_onehot)
        
        

        return {
            "control_mel": control_mel,
            "reference_mel": reference_mel,
            "technique_mel": technique_mel,
            "label": technique_onehot
        }
    
    
    def get_data(self):
        return self.data_pairs
    
    def validate_pairs(self):
        """
        Validate all pairs in self.data_pairs:
            1. Control and target have the same song and clip name (ignoring parent folder).
            2. Reference and target have the same technique.

        Returns:
            bool: True if all pairs are valid, otherwise raises an error.
        """
        for idx, (control_path, reference_path, target_path, _) in enumerate(self.data_pairs):
            # Extract song names and clip names
            control_song = os.path.basename(os.path.dirname(os.path.dirname(control_path)))  # e.g., "不再见"
            target_song = os.path.basename(os.path.dirname(os.path.dirname(target_path)))    # e.g., "不再见"

            control_clip = os.path.basename(control_path)  # e.g., "0000.wav"
            target_clip = os.path.basename(target_path)    # e.g., "0000.wav"

            # Check 1: Control and target are from the same song and same clip
            if control_song != target_song or control_clip != target_clip:
                raise ValueError(
                    f"Control and target mismatch at index {idx}:\n"
                    f"Control: {control_path}\nTarget: {target_path}"
                )

            # Check 2: Reference and target have the same technique
            # Technique is assumed to be encoded in the parent folder (e.g., "Breathy_Group" or "Mixed_Voice_Group")
            reference_technique = os.path.basename(os.path.dirname(reference_path))  # e.g., "Breathy_Group"
            target_technique = os.path.basename(os.path.dirname(target_path))        # e.g., "Breathy_Group"

            if reference_technique != target_technique:
                raise ValueError(
                    f"Reference and target technique mismatch at index {idx}:\n"
                    f"Reference: {reference_path} ({reference_technique})\n"
                    f"Target: {target_path} ({target_technique})"
                )
                
            # print(f"control: {control_song}/{control_clip}, target: {target_song}/{target_clip}")
            # print(f"reference_technique: {reference_technique}, target_technique:{target_technique}")

        print("All pairs are valid: Control and target match, reference and target share the same technique.")
        return True
    
