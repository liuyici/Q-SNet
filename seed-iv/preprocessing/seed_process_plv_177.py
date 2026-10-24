import os
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, hilbert

# 设定路径
root_path = 'E:/Research/EEGDataSet/SEED1/SEED/seed_save/'
save_path = 'E:/Research/EEGDataSet/SEED1/SEED/data_plv/'
os.makedirs(save_path, exist_ok=True)

# 设定基本参数
fixed_length = 177  # 固定片段数
fs = 200  # 采样率是200Hz

# 5个频段
freq_bands = {
    'delta': (1, 3),
    'theta': (4, 7),
    'alpha': (8, 13),
    'beta': (14, 30),
    'gamma': (31, 50)
}

# 设计滤波器
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# 计算PLV
def compute_plv(data):
    """data: shape (n_channels, n_times)"""
    n_channels = data.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels))

    # Hilbert变换
    analytic_signal = hilbert(data, axis=-1)
    phase_data = np.angle(analytic_signal)

    # 两两通道间计算PLV
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            phase_diff = phase_data[i] - phase_data[j]
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / phase_diff.shape[-1])
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv  # 对称

    # 每个通道取它和其他通道平均PLV
    avg_plv = np.sum(plv_matrix, axis=1) / (n_channels - 1)
    return avg_plv  # shape: (n_channels,)

# 主循环
for i in range(15):  # 15 subjects
    for j in range(3):  # 3 sessions
        one_session = []
        one_session_label = []

        for k in range(15):  # 15 trials
            trial_tmp = scipy.io.loadmat(root_path + 'S%d_%d_%d.mat' % (i+1, j+1, k+1))
            trial_data = trial_tmp['trial_data']  # (62, 时间点数)
            trial_label = int(np.squeeze(trial_tmp['trial_label']))  # 0,1,2

            trial_number = trial_data.shape[1] // 200  # 2秒一段，400点

            if trial_number < fixed_length:
                continue  # 太短跳过

            # 取中间的fixed_length个片段
            start_idx = (trial_number - fixed_length) // 2
            one_trial = []

            for tmp_num in range(fixed_length):
                segment = trial_data[:, (start_idx + tmp_num) * 200 : (start_idx + tmp_num + 1) * 200]  # (62, 200)
                
                # 每个频段都计算一次PLV
                segment_plv_features = []
                for band_name, (low_f, high_f) in freq_bands.items():
                    filtered = bandpass_filter(segment, low_f, high_f, fs)
                    avg_plv = compute_plv(filtered)  # (62,)
                    segment_plv_features.append(avg_plv)

                segment_plv_features = np.stack(segment_plv_features, axis=-1)  # (62, 5)
                one_trial.append(segment_plv_features)

            one_session.append(one_trial)  # (185, 62, 5)
            one_session_label.append([trial_label] * fixed_length)

        # 转为numpy array
        data = np.array(one_session)  # shape: (若干trial, 185, 62, 5)
        label = np.array(one_session_label)  # shape: (若干trial, 185)

        # 保存
        np.save(save_path + 'S%d_session%d_plv.npy' % (i+1, j+1), data)
        np.save(save_path + 'S%d_session%d_label.npy' % (i+1, j+1), label)

        print('Finished Subject %d Session %d' % (i+1, j+1))
