# Students	Refael  Greenfeld    Danit   Yshaayahu  305030868   312434269
import sys
from collections import defaultdict


def printify(num_output, answer):
    return f"#Output{num_output}\t{answer}\n"


def clean_lines(lines):
    return lines[2::4]


def parse_file(dev_f):
    res = []
    with open(dev_f) as f:
        lines = f.readlines()
        lines = clean_lines(lines)
        lines = map(lambda x: x.strip().split(), lines)
    for l in lines:
        res.extend(l)
    return res


def vocab_stt(corpus: list) -> defaultdict:
    d = defaultdict(int)
    for w in corpus:
        d[w] += 1
    return d


def total_event_num(corpus):
    total = 0
    for s in corpus.values():
        total += s
    return total


def lidstone(word, set_stat: defaultdict, lamb: float) -> float:
    """

    :param word:
    :param set_stat:
    :param lamb:
    :return:
    """
    all_events = total_event_num(set_stat)
    word_events = set_stat[word]
    vocab_size = len(set_stat)
    return (word_events + lamb) / (all_events + (lamb * vocab_size))


def main():
    VOCAB_SIZE = 300000
    dev_f = sys.argv[1]
    test_f = sys.argv[2]
    input_word = sys.argv[3]
    output_f = sys.argv[4]
    res = init_res(dev_f, input_word, output_f, test_f)

    dev = parse_file(dev_f)
    dev_stat = vocab_stt(dev)

    o5 = VOCAB_SIZE
    res.append(printify(5, o5))

    o6 = 1 / VOCAB_SIZE
    res.append(printify(6, o6))

    o7 = total_event_num(dev_stat)
    res.append(printify(7, o7))

    training_set = dev[:int(0.9 * len(dev))]
    training_stat = vocab_stt(training_set)

    validation_set = dev[int(0.9 * len(dev)):]
    validation_stat = vocab_stt(validation_set)

    o8 = total_event_num(validation_stat)
    o9 = total_event_num(training_stat)

    res.append(printify(8, o8))
    res.append(printify(9, o9))

    o10 = len(training_stat)
    res.append(printify(10, o10))

    o11 = training_stat[input_word]
    res.append(printify(11, o11))

    mle_input = o11 / o9
    o12 = mle_input
    res.append(printify(12, o12))

    o13 = training_stat["unseen-word"]
    res.append(printify(13, o13))

    o14 = lidstone(input_word, training_stat, 0.1)
    o15 = lidstone("unseen-word", training_stat, 0.1)
    res.append(printify(14, o14))
    res.append(printify(15, o15))

    # preplexity()

    print_output(output_f, res)

    # output5 = voc_size(dev_f)


def print_output(output_f, res):
    with open(output_f, mode='w') as f:
        f.writelines(res)


def init_res(dev_f, input_word, output_f, test_f):
    return ["#Students	Refael Greenfeld	Danit Yshaayahu 305030868	312434269\n",
            printify(1, dev_f),
            printify(2, test_f),
            printify(3, input_word),
            printify(4, output_f)]


if __name__ == '__main__':
    # parse_file("datasetsss/datasetsss/develop.txt")
    main()
