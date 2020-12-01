# Students	Refael  Greenfeld    Danit   Yshaayahu  305030868   312434269
import sys
import math
from collections import defaultdict

DEBUG = True
VOCAB_SIZE = 300000


def printify(num_output, answer):
    return f"#Output{num_output}\t{answer}\n"


def clean_lines(lines):
    return lines[2::4]


def perplexity_lidstone(set_stat: defaultdict, train_stat: defaultdict, lamb, num_all_events):
    """
    Calculate 2^ -(sum(p(w \in W)/ |H|) when using lidstone to calculate probabilities
    :param set_stat: The set evaluating on
    :param lamb: the lambda value to use in lidstone
    :param train_stat: data of the trained model
    :param num_all_events: number of events in train
    :return: the perplexity of the model
    """
    log_sum = 0
    for w in set_stat.keys():
        log_sum += log2(lidstone(w, train_stat, lamb, num_all_events)) * set_stat[w]
    prep_score = 2 ** (-(log_sum / total_event_num(set_stat)))
    return round(prep_score, 2)


def perplexity_held_out(h_stat, h_size, test_stat, train_stat):
    """
    Calculate 2^ -(sum(p(w \in W)/ |H|) when using held-out to calculate probabilities
    :param h_stat: H word count
    :param h_size: how many "events" there is in H
    :param test_stat: the test word count
    :param train_stat: the train word count
    :return: the perplexity of the model
    """
    t_reverse_stat = create_rev_stat(train_stat)
    log_sum = 0
    for w in test_stat.keys():
        log_sum += log2(held_out_estimation(w, h_stat, h_size, train_stat, t_reverse_stat)) * test_stat[w]
    prep_score = 2 ** (-(log_sum / total_event_num(test_stat)))
    return round(prep_score, 2)


def log2(n):
    """
    Log 2 with 0 defined as 0
    :param n: the num to log
    :return: log2
    """
    if n == 0:
        return 0
    else:
        return math.log2(n)


def parse_file(input_f):
    res = []
    with open(input_f) as f:
        lines = f.readlines()
        lines = clean_lines(lines)
        lines = map(lambda x: x.strip().split(), lines)
    for l in lines:
        res.extend(l)
    return res


def vocab_stt(corpus: list) -> defaultdict:
    """
    Calculate teh corpus word stat appearances
    :param corpus:
    :return: the corpus word stat
    """
    d = defaultdict(int)
    for w in corpus:
        d[w] += 1
    return d


def total_event_num(corpus: defaultdict) -> int:
    return sum(list(corpus.values()))


def lidstone(word, set_stat: defaultdict, lamb: float, num_all_events) -> float:
    """
    Calculate lidstone estimation using a given lambda -> (c(x) + lambda) / (|S| + lambda|X|)
    :param word: the word to Calculate lidstone estimation on
    :param set_stat: the set stat
    :param lamb: lambda
    :param num_all_events: size of all set
    :return: the lidstone estimation for the word
    """
    word_events = set_stat[word]
    return (word_events + lamb) / (num_all_events + (lamb * VOCAB_SIZE))


def init_res(dev_f, input_word, output_f, test_f):
    o5 = VOCAB_SIZE
    o6 = 1 / VOCAB_SIZE
    return ["#Students	Refael Greenfeld	Danit Yshaayahu 305030868	312434269\n",
            printify(1, dev_f),
            printify(2, test_f),
            printify(3, input_word),
            printify(4, output_f),
            printify(5, o5),
            printify(6, o6)]


def part2(dev_stat, res):
    o7 = total_event_num(dev_stat)
    res.append(printify(7, o7))

def find_best_lambda(validation_stat, training_stat, num_all_events):
    cur_lambda = 0.01
    min_lambda = 0
    min_perplexity = float("inf")

    while cur_lambda <= 2:
        cur_per = perplexity_lidstone(validation_stat, training_stat, cur_lambda, num_all_events)
        if cur_per < min_perplexity:
            min_perplexity = cur_per
            min_lambda = cur_lambda
        cur_lambda += 0.01

    return round(min_lambda, 2), min_perplexity

def part3(dev, input_word, res):
    training_set = dev[:int(0.9 * len(dev))]
    training_stat = vocab_stt(training_set)
    validation_set = dev[int(0.9 * len(dev)):]
    validation_stat = vocab_stt(validation_set)
    num_all_events = total_event_num(training_stat)

    # Calculate the number of events in the sets
    o8 = total_event_num(validation_stat)
    o9 = total_event_num(training_stat)
    res.append(printify(8, o8))
    res.append(printify(9, o9))

    # Calculate the number of *DIFFERENT* events in the training set
    o10 = len(training_stat)
    res.append(printify(10, o10))

    # The number of times the event INPUT WORD appears in the training set
    o11 = training_stat[input_word]
    res.append(printify(11, o11))

    # The MLE of the input word
    mle_input = o11 / o9
    o12 = mle_input
    res.append(printify(12, o12))

    # The MLE of an unseen word
    o13 = training_stat["unseen-word"]
    res.append(printify(13, o13))

    # Lidstone estimation for lambda 0.1
    o14 = lidstone(input_word, training_stat, 0.1, num_all_events)
    o15 = lidstone("unseen-word", training_stat, 0.1, num_all_events)
    res.append(printify(14, o14))
    res.append(printify(15, o15))

    # Finding the lambda which minimise the perplexity
    o16 = perplexity_lidstone(validation_stat, training_stat, 0.01, num_all_events)
    o17 = perplexity_lidstone(validation_stat, training_stat, 0.1, num_all_events)
    o18 = perplexity_lidstone(validation_stat, training_stat, 1, num_all_events)
    res.append(printify(16, o16))
    res.append(printify(17, o17))
    res.append(printify(18, o18))

    best_lambda, best_per = find_best_lambda(validation_stat, training_stat, num_all_events)

    o19 = best_lambda
    res.append(printify(19, o19))
    o20 = best_per
    res.append(printify(20, o20))

    if DEBUG:
        sum_lidstone(o19, training_stat, num_all_events)

    return best_lambda, training_stat, num_all_events


def sum_lidstone(lambda_v, training_stat, num_all_events):
    """
    Debug the model to verify it sums all the probabilities up to 1
    :param lambda_v: the best lambda
    :param training_stat: training_stat
    :param num_all_events: number of all events in training
    :return: it halt the program if fails
    """
    p_unseen = lidstone("unseen-word", training_stat, lambda_v, num_all_events)
    s = p_unseen * (VOCAB_SIZE - len(training_stat))
    for word in training_stat:
        if training_stat[word] != 0:
            s += lidstone(word, training_stat, lambda_v, num_all_events)

    print(f"sum lidstone: {s}")


def create_rev_stat(set_stat):
    reversed_dict = defaultdict(list)
    for key, value in set_stat.items():
        reversed_dict[value].append(key)
    return reversed_dict


def get_unseen_held_out(h_stat, t_stat):
    """
    Calculate the probability estimation for unseen word using Held out
    """
    # t_0
    unseen_sum = 0
    for w_h in h_stat:
        if t_stat[w_h] == 0:
            unseen_sum += h_stat[w_h]

    return unseen_sum


def held_out_estimation(word, h_stat, h_size, t_stat, t_reverse_stat):
    """
    Calculate the probability estimation for a word using Held out
    """
    r = t_stat[word]
    if r == 0:
        n_r = VOCAB_SIZE - len(t_stat)
        t_r = get_unseen_held_out(h_stat, t_stat)
    else:
        n_r = len(t_reverse_stat[r])
        t_r = get_tr(h_stat, t_reverse_stat, r)

    return t_r / (n_r * h_size)


def part4(dev, input_word, res):
    t = dev[:int(0.5 * len(dev))]
    t_stat = vocab_stt(t)

    h = dev[int(0.5 * len(dev)):]
    h_stat = vocab_stt(h)

    # Number of events in the Held-Out sets
    o21 = total_event_num(t_stat)
    h_size = total_event_num(h_stat)
    o22 = h_size
    res.append(printify(21, o21))
    res.append(printify(22, o22))

    # Held out estimation
    t_reverse_stat = create_rev_stat(t_stat)
    o23 = held_out_estimation(input_word, h_stat, h_size, t_stat, t_reverse_stat)
    o24 = held_out_estimation("unseen-word", h_stat, h_size, t_stat, t_reverse_stat)
    res.append(printify(23, o23))
    res.append(printify(24, o24))

    if DEBUG:
        sum_held_out(h_size, h_stat, t_reverse_stat, t_stat, o24)

    return h_stat, h_size, t_stat, t_reverse_stat


def sum_held_out(h_size, h_stat, t_reverse_stat, t_stat, unseen_ord_prob):
    """
    Debug the model to verify it sums all the probabilities up to 1
    :return: it halt the program uf fails
    """
    s = unseen_ord_prob * (VOCAB_SIZE - len(t_stat))
    for word in t_stat:
        if t_stat[word] != 0:
            s += held_out_estimation(word, h_stat, h_size, t_stat, t_reverse_stat)

    print(f"sum held out: {s}")


def get_tr(h_stat, t_reverse_stat, r):
    t_r = 0
    for w in t_reverse_stat[r]:
        t_r += h_stat[w]
    return t_r


def print_output(output_f, res):
    with open(output_f, mode='w') as f:
        f.writelines(res)


def stringify_table(f_lambda, f_h, n_t_r, t_r):
    res = "\n"
    for r, (a,b,c,d) in enumerate(zip(f_lambda, f_h, n_t_r, t_r)):
        res += f"{r}\t{a}\t{b}\t{c}\t{d}\n"
    return res


def main():
    dev_f = sys.argv[1]
    test_f = sys.argv[2]
    input_word = sys.argv[3]
    output_f = sys.argv[4]
    res = init_res(dev_f, input_word, output_f, test_f)

    dev = parse_file(dev_f)
    dev_stat = vocab_stt(dev)

    part2(dev_stat, res)
    best_lambda, training_stat, num_all_events = part3(dev, input_word, res)
    h_stat, h_size, t_stat, t_reverse_stat = part4(dev, input_word, res)

    # part 6 - Evaluation on test set
    test = parse_file(test_f)
    test_stat = vocab_stt(test)

    # Number of events in test set
    o25 = total_event_num(test_stat)
    res.append(printify(25, o25))

    # perplexity on test
    o26 = perplexity_lidstone(test_stat, training_stat, best_lambda, num_all_events)
    o27 = perplexity_held_out(h_stat, h_size, test_stat, t_stat)
    res.append(printify(26, o26))
    res.append(printify(27, o27))

    # Find best model
    better_model = min([(o26, "L"), (o27, "H")], key=lambda x: x[0])[1]
    o28 = better_model
    res.append(printify(28, o28))

    # Creating table
    create_r_table(h_stat, h_size, t_reverse_stat, t_stat, training_stat, best_lambda, res)

    # And.......DONE!
    print_output(output_f, res)


def create_r_table(h_stat, h_size, t_reverse_stat_ho, t_stat_ho, t_stat_lid, best_lambda, res):
    t_r_0 = get_unseen_held_out(h_stat, t_stat_ho)
    n_t_r = [VOCAB_SIZE - len(t_stat_ho)]
    t_r = [t_r_0]

    t_reverse_stat_lid = create_rev_stat(t_stat_lid)
    all_events_lid = total_event_num(t_stat_lid)
    f_lambda = [round(lidstone("unseen-word", t_stat_lid, best_lambda, all_events_lid), 5)]
    f_h = [round(held_out_estimation("unseen-word", h_stat, h_size, t_stat_ho, t_reverse_stat_ho), 5)]

    for r in range(1, 10):
        n_r = len(t_reverse_stat_ho[r])
        n_t_r.append(n_r)

        t_r.append(round(get_tr(h_stat, t_reverse_stat_ho, r), 5))

        word_r_times_lid = t_reverse_stat_lid[r][0]
        f_lambda_r = lidstone(word_r_times_lid, t_stat_lid, best_lambda, all_events_lid)
        f_lambda.append(round(f_lambda_r, 5))

        word_r_times_ho = t_reverse_stat_ho[r][0]
        f_h_r = held_out_estimation(word_r_times_ho, h_stat, h_size, t_stat_ho, t_reverse_stat_ho)
        f_h.append(round(f_h_r, 5))

    o29 = stringify_table(f_lambda, f_h, n_t_r, t_r)
    res.append(printify(29, o29))


if __name__ == '__main__':
    main()
