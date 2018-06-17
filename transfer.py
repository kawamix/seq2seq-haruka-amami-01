import chainer
from chainer import links as L
from chainer import functions as F
from chainer import optimizers
from collections import Counter
import numpy as np
from chainer import cuda
import random
import math
from datetime import datetime
import yaml
from mecab_tokenizer import Tokenizer
import util

from logging import getLogger, StreamHandler, INFO, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)


class EncoderDecoder(chainer.Chain):
    def __init__(self, encoder_layers, decoder_layers, input_vocab_size, output_vocab_size, embed_size,
                 hidden_size, dropout, ARR):
        super(EncoderDecoder, self).__init__()
        with self.init_scope():
            self.embed_input = L.EmbedID(in_size=input_vocab_size, out_size=embed_size)
            self.embed_output = L.EmbedID(in_size=output_vocab_size, out_size=embed_size)
            self.encoder = L.NStepLSTM(n_layers=encoder_layers, in_size=embed_size, out_size=hidden_size,
                                       dropout=dropout)
            self.decoder = L.NStepLSTM(n_layers=decoder_layers, in_size=embed_size, out_size=hidden_size,
                                       dropout=dropout)
            self.output = L.Linear(hidden_size, output_vocab_size)
            self.ARR = ARR

    def __call__(self, input, output):
        input = [x[::-1] for x in input]
        batch = len(input)
        eos = self.ARR.array([EOS_TAG[1]], dtype=self.ARR.int32)
        y_in = [F.concat([eos, y], axis=0) for y in output]
        y_out = [F.concat([y, eos], axis=0) for y in output]
        input_embed = [self.embed_input(i) for i in input]
        output_embed = [self.embed_output(y) for y in y_in]
        e_hx, e_cx, _ = self.encoder(cx=None, xs=input_embed, hx=None)
        _, _, os = self.decoder(cx=e_cx, xs=output_embed, hx=e_hx)
        op = self.output(F.concat(os, axis=0))
        loss = F.sum(F.softmax_cross_entropy(op, F.concat(y_out, axis=0), reduce='no'))
        return loss / batch

    def test(self, input, max_length):
        input = input[::-1]
        input_embed = [self.embed_input(input)]
        e_hx, e_cx, _ = self.encoder(cx=None, xs=input_embed, hx=None)
        ys = self.ARR.array([EOS_TAG[1]], dtype=self.ARR.int32)
        result = []
        for i in range(max_length):
            output_embed = [self.embed_output(ys)]
            e_hx, e_cx, ys = self.decoder(cx=e_cx, xs=output_embed, hx=e_hx)
            cys = F.concat(ys, axis=0)
            os = self.output(cys)
            ys = self.ARR.argmax(os.data, axis=1).astype(self.ARR.int32)
            if ys == EOS_TAG[1]:
                break
            result.append(ys)
        # logger.info(result)
        if len(result) > 0:
            result = cuda.to_cpu(self.ARR.concatenate([self.ARR.expand_dims(x, 0) for x in result]).T)
            # logger.info(result)
        return result


def get_pre_train_data(PRE_TRAIN_FILE, max_vocab_size):
    """
    事前学習用データを取得
    :param PRE_TRAIN_FILE:
    :param max_vocab_size:
    :return:
    """
    with open(file=PRE_TRAIN_FILE[0], encoding="utf-8") as question_text_file, open(file=PRE_TRAIN_FILE[1],
                                                                                    encoding="utf-8") as answer_text_file:
        answer_lines = answer_text_file.readlines()
        question_lines = question_text_file.readlines()

    vocab = []
    for line1, line2 in zip(answer_lines, question_lines):
        vocab.extend(line1.replace("\t", " ").split())
        vocab.extend(line2.replace("\t", " ").split())

    logger.debug(vocab)
    vocab_counter = Counter(vocab)
    vocab = [v[0] for v in vocab_counter.most_common(max_vocab_size)]  # 語彙制限
    word_to_id = {v: i + 2 for i, v in enumerate(vocab)}
    word_to_id[UNKNOWN_TAG[0]] = UNKNOWN_TAG[1]
    word_to_id[EOS_TAG[0]] = EOS_TAG[1]
    id_to_word = {i: v for v, i in word_to_id.items()}

    train_data = []
    for question_line, answer_line in zip(question_lines, answer_lines):
        if len(question_line) < 1 or len(answer_line) < 1:
            continue
        input_words = [word_to_id[word] if word in word_to_id.keys() else word_to_id[UNKNOWN_TAG[0]] for word in
                       question_line.split()]
        output_words = [word_to_id[word] if word in word_to_id.keys() else word_to_id[UNKNOWN_TAG[0]] for word in
                        answer_line.split()]
        if len(input_words) > 0 and len(output_words):
            train_data.append((input_words, output_words))
    return train_data, word_to_id, id_to_word


def get_style_train_data(STYLE_TRAIN_FILE, PRE_TRAIN_FILE, pre_word_to_id, pre_id_to_word):
    """
    スタイル付与データを取得
    :param STYLE_TRAIN_FILE:
    :param PRE_TRAIN_FILE:
    :param pre_word_to_id:
    :param pre_id_to_word:
    :return:
    """
    style_vocab = []
    pre_vocab = []
    with open(STYLE_TRAIN_FILE, encoding="utf-8") as style_file, open(PRE_TRAIN_FILE[0],
                                                                      encoding="utf-8") as pre_file1, open(
        PRE_TRAIN_FILE[1], encoding="utf-8") as pre_file2:
        style_lines = style_file.readlines()
        pre_lines1 = pre_file1.readlines()
        pre_lines2 = pre_file2.readlines()
        for style_line, pre_line1, pre_line2 in zip(style_lines, pre_lines1, pre_lines2):
            style_vocab.extend(style_line.replace("\t", " ").split())
            pre_vocab.extend(pre_line1.split())
            pre_vocab.extend(pre_line2.split())
        pre_lines1 = pre_lines2 = None

    # 単語の入れ替え
    style_high_freq_words = restrict_dict(pre_vocab, style_vocab)
    for i, style_high_freq_word in enumerate(style_high_freq_words):
        pre_id_to_word[i + 2 + max_vocab_size - top_vocab_size] = style_high_freq_word
        pre_word_to_id[style_high_freq_word] = i + 2 + max_vocab_size - top_vocab_size

    train_data = []
    for line in style_lines:
        spl = line.split("\t")
        if len(spl) < 2:
            continue
        input = spl[0]
        output = spl[1]
        input_words = [pre_word_to_id[word] if word in pre_word_to_id.keys() else pre_word_to_id[UNKNOWN_TAG[0]] for
                       word in input.split()]
        output_words = [pre_word_to_id[word] if word in pre_word_to_id.keys() else pre_word_to_id[UNKNOWN_TAG[0]] for
                        word in output.split()]
        if len(input_words) > 0 and len(output_words):
            train_data.append((input_words, output_words))
    return train_data, pre_word_to_id, pre_id_to_word


def restrict_dict(pre_vocab, style_vocab):
    # 転移学習する側の辞書のみに含まれるN単語を取得
    style_counter = Counter(style_vocab).items()
    pre_counter = Counter(pre_vocab).most_common(max_vocab_size)
    pre_rpl_vocab = [w for w, f in pre_counter[:-top_vocab_size]]
    style_high_freq_words = set()
    for w, f in style_counter:
        if w not in pre_rpl_vocab:
            style_high_freq_words.add(w)
        if len(style_high_freq_words) == top_vocab_size:
            break
    return style_high_freq_words


def train(model, train_data, word_to_id, id_to_word, model_path, mode):
    """
    学習
    :param model:
    :param train_data: 学習データ
    :param word_to_id: 単語 -> 単語ID への変換用辞書
    :param id_to_word: 単語ID -> 単語 への変換用辞書
    :param model_path: モデルの保存先のパス
    :param mode: 学習モード。ただ表示するだけ。
    :return:
    """
    logger.info(word_to_id)
    vocab_size = len(id_to_word)
    logger.info("========== MODE: %s ==========" % mode)
    logger.info("========== vocab_size: %s ==========" % vocab_size)
    logger.info("========== train_data_size:%s ==========" % len(train_data))
    logger.debug(id_to_word)
    util.save(word_to_id, model_path + "_w2i.pkl")
    util.save(id_to_word, model_path + "_i2w.pkl")

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(10.0))
    for epoch in range(epoch_num):
        if GPU:
            gpu_device = 0
            cuda.get_device(gpu_device).use()
            model.to_gpu(gpu_device)
        total_loss = 0
        logger.info("=============== EPOCH {} {} ===============".format(epoch + 1, datetime.now()))
        length = math.ceil(len(train_data) / batch_size)
        logger.debug(train_data)
        for i in range(length):
            logger.debug("===== {} / {} =====".format(i, length))
            mini_batch = train_data[i * batch_size: (i + 1) * batch_size]
            t = [xp.asarray(data[1], dtype=xp.int32) for data in mini_batch]
            x = [xp.asarray(data[0], dtype=xp.int32) for data in mini_batch]
            logger.debug("x:{} , y:{}".format(x, t))
            loss = model(x, t)
            total_loss += loss.data
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            logger.debug("=============== loss: %s ===============" % loss)
        logger.info("=============== total_loss: %s ===============" % total_loss)
        test_input = xp.array(train_data[random.randint(0, len(train_data) - 1)][0], dtype=xp.int32)
        test_result = model.test(test_input, SEQUENCE_MAX_LENGTH)
        logger.info("=============== TEST ===============")
        logger.info("input: %s" % str([id_to_word.get(int(id_), UNKNOWN_TAG[0]) for id_ in test_input]))
        logger.debug(test_result)
        logger.info("output: %s" % str([id_to_word.get(int(id_), UNKNOWN_TAG[0]) for id_ in test_result[0]]))
        model.to_cpu()
        chainer.serializers.save_npz(model_path, model)


def pre_train(PRE_TRAIN_FILE, PRE_MODEL_FILE):
    """
    事前学習
    :return:
    """
    pre_train_data, pre_word_to_id, pre_id_to_word = get_pre_train_data(PRE_TRAIN_FILE, max_vocab_size)
    vocab_size = len(pre_word_to_id)
    pre_train_model = EncoderDecoder(encoder_layers=lstm_layers, decoder_layers=lstm_layers,
                                     input_vocab_size=vocab_size, output_vocab_size=vocab_size, embed_size=embed_size,
                                     hidden_size=hidden_size, dropout=dropout, ARR=xp)
    train(model=pre_train_model, train_data=pre_train_data, word_to_id=pre_word_to_id, id_to_word=pre_id_to_word,
          model_path=PRE_MODEL_FILE, mode="PRE_TRAIN")
    return pre_word_to_id, pre_id_to_word


def transfer_train(PRE_TRAIN_FILE, STYLE_TRAIN_FILE, STYLE_MODEL_FILE, PRE_MODEL_FILE, pre_word_to_id, pre_id_to_word):
    """
    転移学習
    :return:
    """
    style_train_data, style_word_to_id, style_id_to_word = get_style_train_data(STYLE_TRAIN_FILE, PRE_TRAIN_FILE,
                                                                                pre_word_to_id, pre_id_to_word)
    vocab_size = len(style_id_to_word)
    pre_train_model = EncoderDecoder(encoder_layers=lstm_layers, decoder_layers=lstm_layers,
                                     input_vocab_size=vocab_size, output_vocab_size=vocab_size, embed_size=embed_size,
                                     hidden_size=hidden_size, dropout=dropout, ARR=xp)
    chainer.serializers.load_npz(PRE_MODEL_FILE, pre_train_model)
    train(model=pre_train_model, train_data=style_train_data, word_to_id=style_word_to_id, id_to_word=style_id_to_word,
          model_path=STYLE_MODEL_FILE, mode="STYLE_TRAIN")


def tokenize():
    """
    入力に使うデータをすべて形態素解析
    :return:
    """
    logger.info("========= START TO TOKENIZE ==========")
    tokenizer = Tokenizer("-Ochasen -d " + neologd_dic_path)
    # 事前学習用データの形態素解析
    pre_train_tokenized = []
    with open(PRE_TRAIN_FILE[0], encoding="utf-8") as f1, open(PRE_TRAIN_FILE[1], encoding="utf-8") as f2:
        for line1, line2 in zip(f1.readlines(), f2.readlines()):
            if len(line1) < 1 or len(line2) < 1:
                continue
            try:
                pre_train_tokenized.append((" ".join([token.surface for token in tokenizer.tokenize(line1)]),
                                            " ".join([token.surface for token in tokenizer.tokenize(line2)])))
            except:  # 形態素解析でエラー発生の場合
                pass
    with open(tmp_dir + "pretrain_input.txt", "wt", encoding="utf-8") as f1, open(tmp_dir + "pretrain_output.txt", "wt",
                                                                                  encoding="utf-8") as f2:
        for line1, line2 in pre_train_tokenized:
            f1.write(line1 + "\r\n")
            f2.write(line2 + "\r\n")
    pre_train_tokenized = None

    # 転移学習用データの形態素解析
    style_train_tokenized = []
    with open(STYLE_TRAIN_FILE, encoding="utf-8") as f:
        for line in f.readlines():
            spl = line.split("\t")
            if len(spl) < 2 or len(spl[0]) < 1 or len(spl[1]) < 1:
                continue
            try:
                style_train_tokenized.append(
                    (" ".join([token.surface for token in tokenizer.tokenize(spl[0])])) + "\t" +
                    " ".join([token.surface for token in tokenizer.tokenize(spl[1])]) + "\r\n")
            except:
                pass
    with open(tmp_dir + "style_data.txt", "wt", encoding="utf-8") as f:
        f.writelines(style_train_tokenized)
    logger.info("========= FINISH TO TOKENIZE ==========")


def start():
    # まず形態素解析
    # tokenize()
    # 事前学習
    pre_word_to_id, pre_id_to_word = pre_train((tmp_dir + "pretrain_input.txt", tmp_dir + "pretrain_output.txt"),
                                               PRE_MODEL_FILE)
    # 転移学習
    transfer_train((tmp_dir + "pretrain_input.txt", tmp_dir + "pretrain_output.txt"), tmp_dir + "style_data.txt",
                   STYLE_MODEL_FILE, PRE_MODEL_FILE, pre_word_to_id, pre_id_to_word)


class Predict:
    """
    学習済みモデルを用いた予測に使うクラス
    """
    def __init__(self, model_path):
        self.word_to_id = util.load(model_path + "_w2i.pkl")
        self.id_to_word = util.load(model_path + "_i2w.pkl")
        vocab_size = len(self.id_to_word)
        model = EncoderDecoder(encoder_layers=lstm_layers, decoder_layers=lstm_layers,
                               input_vocab_size=vocab_size, output_vocab_size=vocab_size,
                               embed_size=embed_size,
                               hidden_size=hidden_size, dropout=dropout, ARR=np)
        chainer.serializers.load_npz(model_path, model)
        self.model = model

    def predict(self, input_words, maxlength):
        input = np.asarray([self.word_to_id.get(w, UNKNOWN_TAG[1]) for w in input_words], dtype=np.int32)
        y = self.model.test(input, maxlength)
        return [self.id_to_word.get(id, UNKNOWN_TAG[0]) for id in y[0]]


TRAIN = False
GPU = False
xp = cuda.cupy if GPU else np

SEQUENCE_MAX_LENGTH = 50

config = yaml.load(open("config.yml", encoding="utf-8"))
PRE_TRAIN_FILE = (config["pre_train_file"]["q"], config["pre_train_file"]["a"])
STYLE_TRAIN_FILE = config["style_train_file"]
PRE_MODEL_FILE = config["encdec"]["pre_model"]
STYLE_MODEL_FILE = config["encdec"]["style_model"]
epoch_num = int(config["encdec"]["epoch"])
batch_size = int(config["encdec"]["batch"])
embed_size = int(config["encdec"]["embed"])
hidden_size = int(config["encdec"]["hidden"])
dropout = float(config["encdec"]["dropout"])
lstm_layers = int(config["encdec"]["lstm_layers"])
max_vocab_size = int(config["encdec"]["max_vocab_size"])
top_vocab_size = int(config["encdec"]["replaced_vocab_size"])
neologd_dic_path = config["neologd_dic_path"]
tmp_dir = config["tmp_dir"]
UNKNOWN_TAG = ("<UNK>", 0)
EOS_TAG = ("<EOS>", 1)

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 0:
        TRAIN = bool(sys.argv[0])
    if TRAIN:
        start()
    else:
        tokenizer = Tokenizer("-Ochasen -d " + neologd_dic_path)
        predictor = Predict(STYLE_MODEL_FILE)
        while True:
            input_ = input("INPUT>>>")
            output = "".join(
                predictor.predict([token.surface for token in tokenizer.tokenize(input_)], SEQUENCE_MAX_LENGTH))
            print("OUTPUT>>>%s" % output)
