from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
import datetime
import re

# from scipy.io import loadmat
# windows
from state_tdrs import StateTDRS
from beam_search import beam_search




class TDRS(object):
    NAME = 'tdrs'  # Tracking and data relay satellite

    ADJUST = 1 / 144
    REC = 1 / 360
    FINISHING_RATE = 1.0

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TDRSDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTDRS.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TDRS.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    task, virtual, *args = args
    grid_size = 1
    if len(args) > 0:
        task_types, window_types, grid_size = args
    return {
        'task': torch.tensor(task, dtype=torch.float) / grid_size,
        'virtual': torch.tensor(virtual, dtype=torch.float) / grid_size
    }


def conflict(a, b, c, d):
    flag = 1
    if a >= d:
        flag = 0
    elif c >= b:
        flag = 0
    return flag


def match_drs(task, access, num_drs):
    ADJUST_REC = 1 / 144
    ta = torch.zeros((task.shape[0], 2 * num_drs + 2))
    for i in range(task.shape[0]):
        est = task[i, 0]
        let = task[i, 1]
        t = task[i, 2]
        sc = task[i, 3]
        pi = task[i, 4]
        for j in range(access.shape[0]):
            if access[j, 3] == sc:
                start = access[j, 0] - 738188
                endt = access[j, 1] - 738188
                T_1 = max(start, est)
                T_2 = min(endt, let)
                rs = int(access[j, 2])
                if T_2 - T_1 >= t:
                    ta[i, (rs - 1) * 2] = T_1
                    ta[i, 2 * rs - 1] = T_2 - t
                    ta[i, 2 * num_drs] = t + ADJUST_REC
                    ta[i, 2 * num_drs + 1] = pi
    return ta


def generation(data_file, task_number, task_emer, start, end, mea, std, mea2, std2, sc, num_drs):
    task = np.zeros((task_number, 5))
    for i in range(task_number):
        while True:
            a = np.random.uniform(0, 1) * (end - start)
            b = a + np.random.normal(mea, std, 1)[0]
            if b <= end:
                task[i, 0] = a
                task[i, 1] = b
                break
        task[i, 2] = np.random.normal(mea2, std2, 1)[0]
        task[i, 3] = np.random.randint(1, high=sc)
        if i < task_emer:
            task[i, 4] = np.random.uniform(1, 2)
        else:
            task[i, 4] = np.random.uniform(0, 1)
    f = open(data_file, "rb")
    access = pickle.load(f)
    access = access[access[:, 3] <= sc]
    TASK = match_drs(task, access, num_drs=num_drs)

    return TASK


class TDRSDataset(Dataset):

    def __init__(self, filename=None, size=200, size_emer=50, num_samples=1000000, offset=0, data_file=None, num_drs=6,
                 num_us=10):
        super(TDRSDataset, self).__init__()

        self.data_set = []
        self.num_drs = num_drs
        self.num_samples = num_samples
        self.size = size
        self.size_emer = size_emer
        self.data_file = data_file
        self.num_us = num_us
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                a = data[offset:offset + num_samples]
            self.data = a

        else:
            self.data = self.ren()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def ren(self):
        Task = generation(self.data_file, self.size + self.size_emer, self.size_emer, 738188, 738188.5, 1 / 36, 1 / 288,
                          1 / 96,
                          1 / 720, sc=self.num_us,
                          num_drs=self.num_drs)
        Task = torch.unsqueeze(Task, 0)
        for i in range(self.num_samples):
            task = generation(self.data_file, self.size + self.size_emer, self.size_emer, 738188, 738188.5, 1 / 36,
                              1 / 288, 1 / 96,
                              1 / 720,
                              sc=self.num_us,
                              num_drs=self.num_drs)
            task = torch.unsqueeze(task, 0)
            print(i)
            Task = torch.cat((Task, task), 0)

        return Task


if __name__ == '__main__':
    common_size = [200]
    emergent = [50]
    sample = 100
    data_name = ["7_10.txt"]
    us = [10]
    for i in range(len(us)):
        emergent_size = emergent[0]
        filename = data_name[0]
        pattern = r"(\d+)_(\d+).txt"
        match = re.search(pattern, filename)
        if match:
            drs = int(match.group(1))
            num_us = int(match.group(2))
        num_us = us[i]
        commom_task_size = common_size[0]
        data = TDRSDataset(size=commom_task_size, size_emer=emergent_size, num_samples=sample, data_file=filename,
                           num_drs=drs, num_us=num_us)
        # 需要修改generation中的sc！！
        filename = "drs" + str(drs) + "_Us" + str(num_us) + "_task" + str(commom_task_size) + "-" + str(
            emergent_size) + "_total" + str(
            sample) + ".pt"
        torch.save(data, filename)
    print(1)
