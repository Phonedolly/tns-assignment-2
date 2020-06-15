import numpy as np
import matplotlib.pyplot as plt
import json
import constants as const
import tools

if __name__ == '__main__':
    sample = tools.get_sample()  # 정규분포를 따르는 표본들의 배열을 가져온다
    sample_mean = np.zeros((const.NUM_OF_EXPERIMENTS,))  # 각 실험마다의 평균 기록
    sample_variance = np.zeros((const.NUM_OF_EXPERIMENTS,))  # 각 실험마다의 분산 기록

    for i in range(const.NUM_OF_EXPERIMENTS):
        sample_mean[i] = sample[i].mean()
        sample_variance[i] = sample[i].var(ddof=1)

    ''' 데이터를 json으로 출력 '''
    data = {'sample': sample, 'sample mean': sample_mean, 'sample variance': sample_variance}
    with open('var_exp_data.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent='\t', cls=tools.NumpyEncoder)

    ''' 히스토그램 출력 '''
    plt.hist(sample_variance, rwidth=0.9, color='xkcd:hot pink')
    plt.title("Histogram of Sample Variances")
    plt.xlabel("classes")
    plt.ylabel("num of variances in each class")
    plt.show()
