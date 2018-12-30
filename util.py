import tensorflow as tf
import numpy as np
import re
from collections import Counter
import string


def get_record_parser(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(
            features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
            features["ques_idxs"], tf.int32), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(
            features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(
            features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        y1 = tf.reshape(tf.decode_raw(
            features["y1"], tf.float32), [para_limit])
        y2 = tf.reshape(tf.decode_raw(
            features["y2"], tf.float32), [para_limit])
        qa_id = features["id"]
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id
    return parse

def get_record_parser_SemEval(config, is_test=False):
    def parse(example):
        ans_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "ans_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "ans_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "y": tf.FixedLenFeature([], tf.string),
                                               
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(
            features["ans_idxs"], tf.int32), [ans_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
            features["ques_idxs"], tf.int32), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(
            features["ans_char_idxs"], tf.int32), [ans_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(
            features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        y = tf.reshape(tf.decode_raw(
            features["y"], tf.float32), [1])
        qa_id = features["id"]
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            buckets_min = [np.iinfo(np.int32).min] + buckets
            buckets_max = buckets + [np.iinfo(np.int32).max]
            conditions_c = tf.logical_and(
                tf.less(buckets_min, c_len), tf.less_equal(c_len, buckets_max))
            bucket_id = tf.reduce_min(tf.where(conditions_c))
            return bucket_id

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)



def aggregate_s0(s0, y, ypred, k=None):
    """
    Generate tuples (s0, [(y, ypred), ...]) where the list is sorted
    by the ypred score.  This is useful for a variety of list-based
    measures in the "anssel"-type tasks.
    """
    ybys0 = dict()
    for i in range(len(s0)):
        try:
            s0is = s0[i].tostring()
        except AttributeError:
            s0is = str(s0[i])
        if s0is in ybys0:
            ybys0[s0is].append((y[i], ypred[i]))
        else:
            ybys0[s0is] = [(y[i], ypred[i])]

    for s, yl in ybys0.items():
        if k is not None:
            yl = yl[:k]
        ys = sorted(yl, key=lambda yy: yy[1], reverse=True)
        yield (s, ys)


def mrr(s0, y, ypred):
    """
    Compute MRR (mean reciprocial rank) of y-predictions, by grouping
    y-predictions for the same s0 together.  This metric is relevant
    e.g. for the "answer sentence selection" task where we want to
    identify and take top N most relevant sentences.
    """
    rr = []
    for s, ys in aggregate_s0(s0, y, ypred):
        if np.sum([yy[0] for yy in ys]) == 0:
            continue  # do not include s0 with no right answers in MRR
        ysd = dict()
        for yy in ys:
            if yy[1][0] in ysd:
                ysd[yy[1][0]].append(yy[0])
            else:
                ysd[yy[1][0]] = [yy[0]]
        rank = 0
        for yp in sorted(ysd.keys(), reverse=True):
            if np.sum(ysd[yp]) > 0:
                rankofs = 1 - np.sum(ysd[yp]) / len(ysd[yp])
                rank += len(ysd[yp]) * rankofs
                break
            rank += len(ysd[yp])
        rr.append(1 / float(1 + rank))
    return np.mean(rr)


def map_(s0, y, ypred):
    MAP = []
    for s, ys in aggregate_s0(s0, y, ypred):
        candidates = ys
        avg_prec = 0
        precisions = []
        num_correct = 0
        for i in range(len(candidates)):
            if candidates[i][0] == 1:
                num_correct += 1
                precisions.append(num_correct / (i + 1))
        if len(precisions):
            avg_prec = sum(precisions) / len(precisions)
        MAP.append(avg_prec)
    return np.mean(MAP)

class AnsSelCB():
    """ A callback that monitors answer selection validation ACC after each epoch """

    def __init__(self, val_q, y, inputs):
        self.val_q = val_q
        self.val_y = y
        self.val_inputs = inputs

    def on_epoch_end(self, pred):
        logs={}
        mrr_ = mrr(self.val_q, self.val_y, pred)
        map__ = map_(self.val_q, self.val_y, pred)
        # print('val MRR %f; MAP: %f' % (mrr_, map__))
        return {"map": map__, "mrr": mrr_}
