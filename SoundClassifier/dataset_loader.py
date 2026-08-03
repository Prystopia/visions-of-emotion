from SoundClassifier.audio_util_load import AudioUtil
from torch.utils.data import Dataset

class SoundsDS(Dataset):
    def __init__(self, df, data_path, isInf):
        self.df = df
        self.data_path = str(data_path)
        self.isInf = isInf
        self.duration = 5000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    def __len__(self):
        if not self.isInf:
            return len(self.df)
        else:
            return 1

    def __getitem__(self, idx):
        if not self.isInf:
            audio_file = self.data_path + self.df.loc[idx, 'relative_path']
            class_id = self.df.loc[idx, 'classID']
        else:
            audio_file = self.data_path
            class_id = -1
        
        aud = self.transformAudio(audio_file)
        return aud, class_id

    def transformAudio(self, file_loc):
        aud = AudioUtil.open(file_loc)
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        if not self.isInf:
            shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
            sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft = 1024, hop_len=None)
            aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
            return aug_sgram
        else:
            sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft = 1024, hop_len=None)
            return sgram