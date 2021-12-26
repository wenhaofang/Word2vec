class HuffmanNode():
    def __init__(self, word_id, word_freq):
        self.word_id = word_id
        self.word_freq = word_freq
        self.lchild = None
        self.rchild = None
        self.huffman_code = []
        self.huffman_path = []

class HuffmanTree():
    def __init__(self , word2freq):
        self.word_num = len(word2freq)
        self.word_id_code = dict()
        self.word_id_path = dict()
        self.huffman_root = None
        self.huffman_tree = [
            HuffmanNode(word_id, word_freq) for word_id, word_freq in word2freq.items()
        ]
        unmerge_node_list = [
            HuffmanNode(word_id, word_freq) for word_id, word_freq in word2freq.items()
        ]
        self.build_tree(unmerge_node_list)
        self.build_code_and_path()

    def merge_node(self, node1, node2):
        parent_node = HuffmanNode(
            len(self.huffman_tree),
            node1.word_freq + node2.word_freq
        )

        if (
            node1.word_freq >= node2.word_freq
        ):
            parent_node.lchild = node1
            parent_node.rchild = node2
        else:
            parent_node.lchild = node2
            parent_node.rchild = node1

        self.huffman_tree.append(parent_node)

        return parent_node

    def build_tree(self, node_list):
        while len (node_list) > 1:
            i1 = 0
            i2 = 1
            if node_list[i2].word_freq < node_list[i1].word_freq:
                _i = i1
                i1 = i2
                i2 = _i
            for i in range(2, len(node_list)):
                if node_list[i].word_freq < node_list[i2].word_freq:
                    i2 = i
                    if node_list[i2].word_freq < node_list[i1].word_freq:
                        _i = i1
                        i1 = i2
                        i2 = _i

            parent_node = self.merge_node(node_list[i1], node_list[i2])

            if i1 == i2:
                raise RuntimeError()
            elif i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1 > i2:
                node_list.pop(i1)
                node_list.pop(i2)

            node_list.insert(0, parent_node)

        self.huffman_root = node_list[0]

    def build_code_and_path(self):
        stack = [self.huffman_root]

        while len(stack) > 0:
            node = stack.pop()

            while node.lchild or node.rchild:
                code = node.huffman_code
                path = node.huffman_path
                node.lchild.huffman_code = code + [1]
                node.rchild.huffman_code = code + [0]
                node.lchild.huffman_path = path + [node.word_id]
                node.rchild.huffman_path = path + [node.word_id]
                stack.append(node.rchild)
                node = node.lchild

            word_code = node.huffman_code
            word_path = node.huffman_path
            self.huffman_tree[node.word_id].huffman_code = word_code
            self.huffman_tree[node.word_id].huffman_path = word_path
            self.word_id_code[node.word_id] = word_code
            self.word_id_path[node.word_id] = word_path

    def get_pos_and_neg_path(self):
        pos_paths = []
        neg_paths = []

        for word_id in range(self.word_num):
            pos_ids = []
            neg_ids = []
            for idx, code in enumerate(self.huffman_tree[word_id].huffman_code):
                if code == 1:
                    pos_ids.append(self.huffman_tree[word_id].huffman_path[idx])
                if code == 0:
                    neg_ids.append(self.huffman_tree[word_id].huffman_path[idx])

            pos_paths.append(pos_ids)
            neg_paths.append(neg_ids)

        return pos_paths, neg_paths
