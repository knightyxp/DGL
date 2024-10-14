import numpy as np
import logging
import argparse

# 初始化参数解析器
parser = argparse.ArgumentParser(description='Search for Best Beta Script')


parser.add_argument("--t2v_k", default=1, type=int)
parser.add_argument("--t2v_beta", default=0.87, type=float)
parser.add_argument("--t2v_theta", default=3, type=float)
parser.add_argument("--v2t_k", default=1, type=int)
parser.add_argument("--v2t_beta", default=0.87, type=float)
parser.add_argument("--v2t_theta", default=5, type=float)
parser.add_argument("--dataset", default="msrvtt", type=str)

args = parser.parse_args()


log_file_path = 'search_best_beta_{}_dgl_transformer_vit16.log'.format(args.dataset)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path, mode='a'),  # 注意这里的模式是 'a'
                        logging.StreamHandler()
                    ])


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['Rsum'] = metrics['R1'] + metrics['R5'] + metrics['R10']
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics



def get_retrieved_videos(sims, k, theta):
    argm = np.argsort(-sims, axis=1)
    topk = argm[:,:k].reshape(-1)
    # retrieved_videos = np.unique(topk)
    # return retrieved_videos
    retrieved_videos, occurrence_count = np.unique(topk, return_counts=True)
    return retrieved_videos[occurrence_count>=theta]

# Returns list of indices to normalize from sims based on videos
def get_index_to_normalize(sims, videos):
    argm = np.argsort(-sims, axis=1)[:,0]
    result = np.array(list(map(lambda x: x in videos, argm)))
    result = np.nonzero(result)
    return result

def qb_norm(train_test, test_test, k, beta, theta):
    retrieved_videos = get_retrieved_videos(train_test, k, theta)
    test_test_normalized = test_test
    train_test = np.exp(train_test*beta)
    test_test = np.exp(test_test*beta)

    normalizing_sum = np.sum(train_test, axis=0)
    index_for_normalizing = get_index_to_normalize(test_test, retrieved_videos)
    test_test_normalized[index_for_normalizing, :] = \
        np.divide(test_test[index_for_normalizing, :], normalizing_sum)
    return test_test_normalized

def search_best_beta(train_test_t2v, train_test_v2t, sim_matrix, beta_range=(0.5, 1.0), beta_step=0.01):
    best_beta = None
    best_r1 = 0.0

    # 循环遍历可能的 beta 值
    for beta in np.arange(beta_range[0], beta_range[1] + beta_step, beta_step):
        # 使用当前的beta进行正则化
        t2v_normalized = qb_norm(train_test_t2v.copy(), sim_matrix.copy(), args.t2v_k, beta, args.t2v_theta)
        v2t_normalized = qb_norm(train_test_v2t.T.copy(), sim_matrix.T.copy(), args.v2t_k, beta, args.v2t_theta)

        # 计算当前 beta 下的 R@1
        tv_metrics = compute_metrics(t2v_normalized)
        vt_metrics = compute_metrics(v2t_normalized)
        
        # return for final logging
        info_str = []
        info_str.append("Text-to-Video:")
        info_str.append(' (metric) >>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
        info_str.append("Video-to-Text:")
        info_str.append(' (metric) >>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
        for info in info_str: logging.info(info)
        curr_r1 = tv_metrics['R1']  
        if curr_r1 > best_r1:
            best_r1 = curr_r1
            best_beta = beta

        # 输出当前 beta 和对应的 R@1 值，方便追踪进度
        logging.info(f'Beta: {beta}, curr R@1: {curr_r1}')

    return best_beta, best_r1

# 加载数据


train_test_t2v = np.load('qb_norm_sim_matrix/{}_vit16_train_test_t2v.npy'.format(args.dataset))
train_test_v2t = np.load('qb_norm_sim_matrix/{}_vit16_train_test_v2t.npy'.format(args.dataset))
sim_matrix = np.load('qb_norm_sim_matrix/{}_vit16_sim_matrix.npy'.format(args.dataset))

# 调用上述函数
best_beta, best_r1 = search_best_beta(train_test_t2v, train_test_v2t, sim_matrix)
logging.info(f"Best beta: {best_beta}, with R@1: {best_r1}")