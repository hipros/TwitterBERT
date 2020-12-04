import torch
import argparse
import gluonnlp

from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from torch.utils.data import DataLoader

from dataset import BERTDataset


class Solver(object):
    def __init__(self, config):
        self.class_num = config.num_class
        self.device = config.cuda
        self.train_txt_path = config.train_txt_path
        self.valid_txt_path = config.valid_txt_path
        self.train_batch_size = config.train_batch_size
        self.valid_batch_size = config.valid_batch_size
        self.max_len = config.max_len

        self.model = None
        self.vocab = None
        self.train_loader = None
        self.valid_loader = None
        self.tokenizer = None
        self.token = None

    def load_model(self):
        self.model, self.vocab = get_pytorch_kobert_model()
        self.tokenizer = get_tokenizer()
        self.token = gluonnlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)

    def load_data(self):
        # 각 파라메터 의미 파악 필요
        train_dataset = BERTDataset(self.train_txt_path, 0, 1, self.token, self.max_len, True, False)
        valid_dataset = BERTDataset(self.valid_txt_path, 0, 1, self.token, self.max_len, True, False)

        self.train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.valid_batch_size)

    def run(self):
        self.load_model()
        self.load_data()

        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KoBERT model classifier finetuning")
    parser.add_argument('--num_class', default=2, type=int,
                        help="Class number for classification")
    parser.add_argument('--cuda', default="cuda" if torch.cuda.is_available() else "cpu", type=str,
                        help="Cuda available")

    parser.add_argument('--train_batch_size', default=64, type=int, help="Train batch size")
    parser.add_argument('--valid_batch_size', default=64, type=int, help="Validation batch size")

    parser.add_argument('--train_txt_path', default="total.txt?dl=1", type=str,
                        help="Train dataset for fine tuning")
    parser.add_argument('--valid_txt_path', default="data/ratings_test.txt?dl=1", type=str,
                        help="Validation dataset for fine tuning")

    parser.add_argument('--max_len', default=64, type=int,
                        help="Max length of one input sequence")
    args = parser.parse_args()
    print(args)

    solver = Solver(args)
    solver.run()
