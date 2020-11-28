# Students	Refael  Greenfeld    Danit   Yshaayahu  305030868   312434269
import sys
import math
from collections import defaultdict
DEBUG = False


def printify(num_output, answer):
    return f"#Output{num_output}\t{answer}\n"


def clean_lines(lines):
    return lines[2::4]


def perplexity_lidstone(set_stat: defaultdict, lamb):
    """
    Calculate 2^ -(sum(p(w \in W)/ |H|) when using lidstone to calculate probabilities
    :param set_stat: The set evaluating on
    :param lamb: the lambda value to use in lidstone
    :return: the perplexity of the model
    """
    log_sum = 0
    for w in set_stat.keys():
        log_sum += log2(lidstone(w, set_stat, lamb))
    prep_score = 2 ** (-(log_sum / len(set_stat)))
    return prep_score


def perplexity_held_out(h_stat, h_size, test_stat):
    """
    Calculate 2^ -(sum(p(w \in W)/ |H|) when using held-out to calculate probabilities
    :param h_stat: H word count
    :param h_size: how many "events" there is in H
    :param test_stat: the test word count
    :return: the perplexity of the model
    """
    t_reverse_stat = create_rev_stat(test_stat)
    log_sum = 0
    for w in test_stat.keys():
        log_sum += log2(held_out_estimation(w, h_stat, h_size, test_stat, t_reverse_stat))
    prep_score = 2 ** (-(log_sum / len(test_stat)))
    return prep_score


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
    total = 0
    for s in corpus.values():
        total += s
    return total


def lidstone(word, set_stat: defaultdict, lamb: float) -> float:
    """
    Calculate lidstone estimation using a given lambda -> (c(x) + lambda) / (|S| + lambda|X|)
    :param word: the word to Calculate lidstone estimation on
    :param set_stat: the set stat
    :param lamb: lambda
    :return: the lidstone estimation for the word
    """
    all_events = total_event_num(set_stat)
    word_events = set_stat[word]
    vocab_size = len(set_stat)
    return (word_events + lamb) / (all_events + (lamb * vocab_size))


def init_res(dev_f, input_word, output_f, test_f):
    vocab_size = 300000
    o5 = vocab_size
    o6 = 1 / vocab_size
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


def part3(dev, input_word, res):
    training_set = dev[:int(0.9 * len(dev))]
    training_stat = vocab_stt(training_set)
    validation_set = dev[int(0.9 * len(dev)):]
    validation_stat = vocab_stt(validation_set)

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

    # Lidstone estimation  for lambda 0.1
    o14 = lidstone(input_word, training_stat, 0.1)
    o15 = lidstone("unseen-word", training_stat, 0.1)
    res.append(printify(14, o14))
    res.append(printify(15, o15))

    # Finding the lambda which minimise the perplexity
    lambda_option = [0.01, 0.1, 1]
    o16 = perplexity_lidstone(validation_stat, lambda_option[0])
    o17 = perplexity_lidstone(validation_stat, lambda_option[1])
    o18 = perplexity_lidstone(validation_stat, lambda_option[2])
    res.append(printify(16, o16))
    res.append(printify(17, o17))
    res.append(printify(18, o18))

    min_tup = min([(o16, lambda_option[0]), (o17, lambda_option[1]), (o18, lambda_option[2])], key=lambda x: x[0])
    best_lambda = min_tup[1]
    o19 = best_lambda
    res.append(printify(19, o19))
    o20 = min_tup[0]
    res.append(printify(20, o20))

    if DEBUG:
        sum_lidstone(o19, training_stat, validation_stat)
    return best_lambda


def sum_lidstone(lambda_v, training_stat, validation_stat):
    """
    Debug the model to verify it sums all the probabilities up to 1
    :param lambda_v: the best lambda
    :param training_stat: training_stat
    :param validation_stat: validation_stat
    :return: it halt the program uf fails
    """
    p_unseen = lidstone("unseen-word", training_stat, lambda_v)
    n_0, _ = get_unseen_held_out(validation_stat, training_stat)
    s = p_unseen * n_0
    for word in training_stat:
        if training_stat[word] != 0:
            s += lidstone(word, training_stat, lambda_v)
    print(s)
    assert round(s, 4) == 1


def create_rev_stat(set_stat):
    reversed_dict = defaultdict(list)
    for key, value in set_stat.items():
        reversed_dict[value].append(key)
    return reversed_dict


def get_unseen_held_out(h_stat, t_stat):
    """
    Calculate the probability estimation for unseen word using Held out
    """
    unseen_sum = 0  # T_0
    unseen_events = 0  # N_0
    for w_h in h_stat:
        if t_stat[w_h] == 0:
            unseen_sum += h_stat[w_h]
            unseen_events += 1
    return unseen_events, unseen_sum


def held_out_estimation(word, h_stat, h_size, t_stat, t_reverse_stat):
    """
    Calculate the probability estimation for a word using Held out
    """
    r = t_stat[word]
    if r == 0:
        n_r, t_r = get_unseen_held_out(h_stat, t_stat)
    else:
        n_r = len(t_reverse_stat[r])
        t_r = get_tr(h_stat, t_reverse_stat, r)

    return t_r / (n_r * h_size)


def part4(dev, input_word, res):
    T = dev[:int(0.5 * len(dev))]
    T_stat = vocab_stt(T)

    H = dev[int(0.5 * len(dev)):]
    H_stat = vocab_stt(H)

    # Number of events in the Held-Out sets
    o21 = total_event_num(T_stat)
    H_size = total_event_num(H_stat)
    o22 = H_size
    res.append(printify(21, o21))
    res.append(printify(22, o22))

    # Held out estimation
    T_reverse_stat = create_rev_stat(T_stat)
    o23 = held_out_estimation(input_word, H_stat, H_size, T_stat, T_reverse_stat)
    o24 = held_out_estimation("unseen-word", H_stat, H_size, T_stat, T_reverse_stat)
    res.append(printify(23, o23))
    res.append(printify(24, o24))

    if DEBUG:
        sum_held_out(H_size, H_stat, T_reverse_stat, T_stat, o24)

    return H_stat, H_size, T_stat, T_reverse_stat


def sum_held_out(H_size, H_stat, T_reverse_stat, T_stat, unseen_ord_prob):
    """
    Debug the model to verify it sums all the probabilities up to 1
    :return: it halt the program uf fails
    """
    n_0, t_0 = get_unseen_held_out(H_stat, T_stat)
    s = unseen_ord_prob * n_0
    for word in T_stat:
        if T_stat[word] != 0:
            s += held_out_estimation(word, H_stat, H_size, T_stat, T_reverse_stat)
    print(f"s: {s}")
    assert round(s, 10) == 1


def get_tr(H_stat, T_reverse_stat, r):
    t_r = 0
    for w in T_reverse_stat[r]:
        t_r += H_stat[w]
    return t_r


def print_output(output_f, res):
    with open(output_f, mode='w') as f:
        f.writelines(res)


def stringify_table(f_lambda, f_h, N_T_R, T_R):
    res = "\n"
    for r, (a,b,c,d) in enumerate(zip(f_lambda, f_h, N_T_R, T_R)):
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
    best_lambda = part3(dev, input_word, res)
    H_stat, H_size, T_stat, T_reverse_stat = part4(dev, input_word, res)

    # part 6 - Evaluation on test set
    test = parse_file(test_f)
    test_stat = vocab_stt(test)

    # Number of events in test set
    o25 = total_event_num(test_stat)
    res.append(printify(25, o25))

    # perplexity on test
    o26 = perplexity_lidstone(test_stat, best_lambda)
    o27 = perplexity_held_out(H_stat, H_size, test_stat)
    res.append(printify(26, o26))
    res.append(printify(27, o27))

    # Find best model
    better_model = min([(o26, "L"), (o27, "H")], key=lambda x: x[0])[1]
    o28 = better_model
    res.append(printify(28, o28))

    # Creating table
    create_r_table(H_stat, T_reverse_stat, T_stat, res)

    # And.......DONE!
    print_output(output_f, res)


def create_r_table(H_stat, T_reverse_stat, T_stat, res):
    N_T_0, T_R_0 = get_unseen_held_out(H_stat, T_stat)
    N_T_R = [N_T_0]
    T_R = [T_R_0]
    for r in range(1, 10):
        n_r = len(T_reverse_stat[r])
        N_T_R.append(n_r)
        t_r = get_tr(H_stat, T_reverse_stat, r)
        T_R.append(t_r)
    # TODO fix f_lambda, f_h
    f_lambda = [111] * 10
    f_h = [777] * 10
    o29 = stringify_table(f_lambda, f_h, N_T_R, T_R)
    res.append(printify(29, o29))


if __name__ == '__main__':
    DEBUG = False
    main()
