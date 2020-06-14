import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import json
import constants as const
import tools

if __name__ == '__main__':
    sample = tools.get_sample()  # 정규분포를 따르는 표본들의 배열을 가져온다
    sample_mean = np.zeros((const.NUM_OF_EXPERIMENTS,))  # 각 실험마다의 평균 기록
    sample_variance = np.zeros((const.NUM_OF_EXPERIMENTS,))  # 각 실험마다의 분산 기록
    t_statistic = np.zeros(const.NUM_OF_EXPERIMENTS)  # 각 실험마다의 t-statistic을 기록

    ''' 각 실험에 대한 sample mean, sample variance, t-statistic 계산 '''
    for i in range(const.NUM_OF_EXPERIMENTS):
        sample_mean[i] = sample[i].mean()
        sample_variance[i] = sample[i].var(ddof=1)
        t_statistic[i] = np.sqrt(const.NUM_OF_SAMPLES) * (sample_mean[i] - const.MEAN) / sample_variance[i]

    ''' 데이터를 json으로 출력 '''
    data = {'samples': sample, 'means': sample_mean, 'vars': sample_variance, 't-stats': t_statistic}
    with open('t_exp_data.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent='\t', cls=tools.NumpyEncoder)

    ''' 히스토그램 세팅 '''
    axis_x = np.linspace(-3 * np.sqrt(const.VAR), 3 * np.sqrt(const.VAR), 100)  # 그래프가 가질 x축
    fig, ax1 = plt.subplots()

    color = 'xkcd:lightgreen'
    ax1.set_xlabel('')
    ax1.set_ylabel("num of variances in each class")
    ax1.hist(t_statistic, histtype='bar', label="t-statistics", color=color, rwidth=0.9)
    ax1.set_ylim(ymin=0)

    ''' t-분포 그래프 세팅 '''
    color = 'xkcd:fuchsia'
    ax2 = ax1.twinx()  # ax1과 x축을 공유하고 y축은 따로 쓴다

    t_dist_graph = sp.stats.t(df=const.NUM_OF_SAMPLES).pdf(axis_x)
    ax2.set_ylabel("t-distribution")
    ax2.plot(axis_x, t_dist_graph, label="t$_{29}$", color=color)
    ax2.set_ylim(ymin=0)

    ''' legend 지정, 나머지 작업 처리 '''
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    plt.legend(lines, labels, loc=0)

    plt.title("Histogram of t-statistics and graph of t-dist")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
