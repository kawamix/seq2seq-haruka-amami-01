import MeCab


class Token:
    def __init__(self, node):
        self.surface = node.surface
        self.__set_pos(node.feature)

    def __set_pos(self, feature):
        pos_list = feature.split(",")
        self.base = pos_list[6]
        self.reading = "" if len(pos_list) < 8 else pos_list[7]
        detail_pos_list = []
        for pos in pos_list:
            if pos == "*":
                break
            detail_pos_list.append(pos)
        self.pos = ",".join(detail_pos_list)

    def __eq__(self, other):
        return self.surface == other.surface and self.pos == other.pos

    def __hash__(self):
        return hash(self.surface + self.pos)

    def __str__(self):
        return self.surface


class Tokenizer:
    def __init__(self, arg="mecabrc"):
        self.t = MeCab.Tagger(arg)

    def __parse(self, node):
        token_list = []
        while node:
            token = Token(node)
            if not token.pos.startswith("BOS/EOS"):
                token_list.append(token)
            node = node.next
        return token_list

    def tokenize(self, sentence):
        self.t.parse("")  # これをしないと正しい結果が返ってこないバグが存在
        node = self.t.parseToNode(sentence)
        return self.__parse(node)


if __name__ == '__main__':
    neologd_dic_path = "/hogehoge/mecab-ipadic-neologd" # お好きな辞書のパス
    test = "夜の水面に飛び交う蛍が流れ星みたいで綺麗なのん。のんのんびより。"
    tokenizer = Tokenizer("-Ochasen -d " + neologd_dic_path)
    token_list = tokenizer.tokenize(test)
    for token in token_list:
        print(token.surface + " : " + token.pos + " : " + token.base + " : " + token.reading)
