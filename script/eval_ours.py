import numpy as np
import matplotlib.pyplot as plt

# from ASVspoof2021/eval-package/eval_metrics.py
def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def read_scores_from_file(file_path):
    with open(file_path, 'r') as file:
        scores = file.read()
    return scores

def plot_det_curve(target_scores, nontarget_scores, save_path="research-project-seminar-ASVspoof/result/det_curve.png"):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)

    plt.figure(figsize=(8, 8))
    plt.plot(far, frr, label='DET Curve', color='blue')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('DET Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    # 画像を保存
    plt.savefig(save_path)
    print(f"DET Curve plot saved at: {save_path}")

if __name__  == "__main__":
    file_path = "research-project-seminar-ASVspoof/result/scores-lfcc-PA-ours.txt"
    scores = read_scores_from_file(file_path)

    # 各行を分割して正常な話者と異なる話者のスコアを取得
    lines = scores.strip().split('\n')
    target_scores = []  # 正常な話者のスコア
    nontarget_scores = []  # 異なる話者のスコア

    for line in lines:
        parts = line.split()
        if parts[0].endswith('_spf00_env20'):
            target_scores.append(float(parts[-1]))
        elif parts[0].endswith('_spf01_env20'):
            nontarget_scores.append(float(parts[-1]))

    # EERと対応する閾値を計算
    eer, threshold = compute_eer(np.array(target_scores), np.array(nontarget_scores))
    plot_det_curve(np.array(target_scores), np.array(nontarget_scores))

    # 結果を出力
    print(f'EER: {eer}, Threshold: {threshold}')

"""
DET Curve plot saved at: det_curve.png
EER: 0.25, Threshold: -2.420936400787838
"""
