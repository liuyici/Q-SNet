import numpy as np
import scipy.io
import os

root_path = 'E:/Research/EEGDataSet/SEED1/SEED/seed_save/'
save_path = 'E:/Research/EEGDataSet/SEED1/SEED/data_cv5fold/'
os.makedirs(save_path, exist_ok=True)

fixed_length = 177  # 固定片段数

for i in range(15):
    for j in range(3):
        one_session = []
        one_session_label = []
        
        for k in range(15):
            trial_tmp = scipy.io.loadmat(root_path + 'S%d_%d_%d.mat' % (i+1, j+1, k+1))
            trial_data = trial_tmp['trial_data']  # shape: (62, 47001)
            trial_label = int(np.squeeze(trial_tmp['trial_label']))  # e.g., 0, 1, 2

            trial_number = trial_data.shape[1] // 200  # how many 2s segments

            if trial_number < fixed_length:
                continue  # 跳过太短的trial

            # 计算从哪里开始取中间的185个片段
            start_idx = (trial_number - fixed_length) // 2  # 计算起始位置

            one_trial = []
            for tmp_num in range(fixed_length):  # 取中间的185个片段
                one_trial.append(trial_data[:, (start_idx + tmp_num) * 200 : (start_idx + tmp_num + 1) * 200])  # (62, 200)

            one_session.append(one_trial)  # 185 x 62 x 200
            one_session_label.append([trial_label] * fixed_length)  # 185

        # 转为 numpy array
        data = np.array(one_session)  # shape: (15, 185, 62, 200)
        label = np.array(one_session_label)  # shape: (15, 185)

        # 保存
        np.save(save_path + 'S%d_session%d.npy'%(i+1, j+1), data)
        np.save(save_path + 'S%d_session%d_label.npy'%(i+1, j+1), label)

        print('Finished Subject %d Session %d' % (i+1, j+1))
