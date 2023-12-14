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

def compute_eer_for_each_speaker(file_path):
    scores = read_scores_from_file(file_path)

    # 各話者ごとにスコアを保存
    speaker_scores = {}
    lines = scores.strip().split('\n')

    for line in lines:
        parts = line.split()
        speaker_id = parts[0].split('_')[0]
        score = float(parts[-1])

        if speaker_id not in speaker_scores:
            speaker_scores[speaker_id] = {'target': [], 'nontarget': []}

        if parts[0].endswith('_spf00_env20'):
            speaker_scores[speaker_id]['target'].append(score)
        elif parts[0].endswith('_spf01_env20'):
            speaker_scores[speaker_id]['nontarget'].append(score)

    # 各話者ごとに EER を計算
    for speaker_id, scores in speaker_scores.items():
        target_scores = np.array(scores['target'])
        nontarget_scores = np.array(scores['nontarget'])

        if len(target_scores) != 0 and len(nontarget_scores) != 0:
            eer, threshold = compute_eer(target_scores, nontarget_scores)
            print(f'Speaker {speaker_id}: EER: {eer}, Threshold: {threshold}')
        else:
            print(f'Speaker {speaker_id}: {len(target_scores), len(nontarget_scores)}')

if __name__  == "__main__":
    file_path = "research-project-seminar-ASVspoof/result/scores-lfcc-PA-ours.txt"
    # 発話者ごとのEER
    compute_eer_for_each_speaker(file_path)

    # 全体のEER
    scores = read_scores_from_file(file_path)
    lines = scores.strip().split('\n')
    target_scores = []  # 正常な話者のスコア
    nontarget_scores = []  # 異なる話者のスコア

    for line in lines:
        parts = line.split()
        if parts[0].endswith('_spf00_env20'):
            target_scores.append(float(parts[-1]))
        elif parts[0].endswith('_spf01_env20'):
            nontarget_scores.append(float(parts[-1]))

    plot_det_curve(np.array(target_scores), np.array(nontarget_scores), save_path="research-project-seminar-ASVspoof/result/det_curve.png")
    eer, threshold = compute_eer(np.array(target_scores), np.array(nontarget_scores))
    print(f'EER: {eer}, Threshold: {threshold}')


"""
全体の結果
DET Curve plot saved at: det_curve.png
EER: 0.25, Threshold: -2.420936400787838

発話者ごとの結果
Speaker spk10: EER: 0.0, Threshold: -4.164542757730331
Speaker spk20: EER: 0.1, Threshold: -2.8111762833268017
Speaker spk30: EER: 0.2, Threshold: -2.4986321838136973
Speaker spk40: EER: 0.3, Threshold: -1.684924964055071
Speaker spk50: EER: 0.0, Threshold: -1.4517488078647034
Speaker spk60: EER: 0.1, Threshold: -2.85419032163945
"""
