import json
import numpy as np
import scipy as sp
import scipy.stats
import quantumrandom as qt
import constants as const


# 정규분포를 따르는 샘플들의 배열을 생성한다
def get_sample():
    sample = np.zeros((const.NUM_OF_EXPERIMENTS, const.NUM_OF_SAMPLES))

    generator = qt.cached_generator()

    for i in range(const.NUM_OF_EXPERIMENTS):
        for j in range(const.NUM_OF_SAMPLES):
            sample[i][j] = sp.stats.norm(loc=const.MEAN, scale=np.sqrt(const.VAR)).ppf(
                qt.randfloat(0, 1, generator))  # 평균이 const.MEAN, 표준편차가 sqrt(const.VAR)

    return sample


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
