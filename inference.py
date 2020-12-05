import torch
import gluonnlp

from kobert.pytorch_kobert import get_pytorch_kobert_model
from classifier import BERTClassifier
from kobert.utils import get_tokenizer


class Converter(object):
    def __init__(self, token, max_len, pad, pair, device):
        self.transform = gluonnlp.data.BERTSentenceTransform(token, max_seq_length=max_len,
                                                             pad=pad, pair=pair)
        self.device = device

    def transform_type(self, val, to_cuda):
        val = torch.tensor(val).long()

        if to_cuda is True:
            val = val.to(self.device)

        val = torch.unsqueeze(val, 0)
        return val

    def convert(self, content):
        token_ids, valid_length, segment_ids = self.transform(content)

        token_ids = self.transform_type(token_ids, to_cuda=True)
        segment_ids = self.transform_type(segment_ids, to_cuda=True)
        valid_length = self.transform_type(valid_length, to_cuda=False)

        return token_ids, valid_length, segment_ids


class Inference(object):
    def __init__(self, save_path='save_model/best_model.pth', max_len=64, pad=True, pair=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.max_len = max_len
        self.pad = pad
        self.pair = pair

        self.bert_model = None
        self.vocab = None
        self.model = None
        self.dropout_rt = None
        self.tokenizer = None
        self.token = None
        self.transform = None
        self.line_converter = None

    def load_model(self):
        self.bert_model, self.vocab = get_pytorch_kobert_model(ctx=self.device)
        self.model = BERTClassifier(self.bert_model, dr_rate=self.dropout_rt).to(self.device)

        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))

        self.tokenizer = get_tokenizer()
        self.token = gluonnlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)

        self.line_converter = Converter(self.token, self.max_len, self.pad, self.pair, self.device)

    def predict(self, content):
        token_ids, valid_length, segment_ids = self.line_converter.convert(content)

        return self.model(token_ids, valid_length, segment_ids)

    def initial_model(self):
        self.load_model()
        self.model.eval()


def get_result(result):
    max_vals, max_indices = torch.max(result, 1)
    return "normal tweet" if max_indices == 0 else "malignity tweet"


if __name__ == '__main__':
    ifr = Inference()

    ifr.initial_model()
    result = ifr.predict('쉿 은밀한 만남 여수출장업소 여수출장안마 여수조건만남 여수콜걸 상담톡 C A L L 9 9 9  여수출장아가씨대박 여수출장업소강추 여수출장아가씨강추대행 여수출장업소대박')

    print(get_result(result))



