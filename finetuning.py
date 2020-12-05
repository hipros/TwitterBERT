import torch
import argparse
import gluonnlp
from tqdm import tqdm

from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from dataset import BERTDataset
from classifier import BERTClassifier


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]

    return train_acc


class Solver(object):
    def __init__(self, config):
        self.class_num = config.num_class
        self.device = config.cuda
        self.train_txt_path = config.train_txt_path
        self.valid_txt_path = config.valid_txt_path
        self.train_batch_size = config.train_batch_size
        self.valid_batch_size = config.valid_batch_size
        self.model_save_path = config.save_path
        self.max_len = config.max_len
        self.dropout_rt = config.dropout_rate
        self.num_epochs = config.train_epochs
        self.get_weights = config.get_weights
        self.learning_rate = 5e-5
        self.warmup_ratio = 0.1
        self.max_grad_norm = 1
        self.log_interval = 200

        self.bert_model = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.vocab = None
        self.train_loader = None
        self.valid_loader = None
        self.tokenizer = None
        self.token = None
        self.max_acc = 0

    def load_model(self):
        self.bert_model, self.vocab = get_pytorch_kobert_model(ctx=self.device)
        self.model = BERTClassifier(self.bert_model, dr_rate=self.dropout_rt).to(self.device)
        if self.get_weights:
            print("get model from pretrained weigths")
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))

        self.tokenizer = get_tokenizer()
        self.token = gluonnlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)

    def load_data(self):
        train_dataset = BERTDataset(self.train_txt_path, 0, 1, self.token, self.max_len, True, False)
        valid_dataset = BERTDataset(self.valid_txt_path, 0, 1, self.token, self.max_len, True, False)

        self.train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.valid_batch_size)

        self.set_train()

    def save_model(self):
        print("model_save")
        torch.save(self.model.state_dict(), self.model_save_path)

    def set_train(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [param for name, param in self.model.named_parameters()
                        if not any(nd in name for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [param for name, param in self.model.named_parameters()
                        if any(nd in name for nd in no_decay)], 'weight_decay': 0.0},
        ]
        num_total_train = len(self.train_loader) * self.num_epochs
        warmup_step = int(num_total_train * self.warmup_ratio)

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                         num_training_steps=num_total_train)

    def train_model(self, epoch):
        train_acc = 0.0
        self.model.train()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length = valid_length
            label = label.long().to(self.device)
            out = self.model(token_ids, valid_length, segment_ids)
            loss = self.criterion(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)

        print("epoch {} train acc {}".format(epoch + 1, train_acc / (batch_id + 1)))

    def valid_model(self, epoch):
        test_acc = 0.0
        self.model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.valid_loader)):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length = valid_length
            label = label.long().to(self.device)
            out = self.model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)

        if self.max_acc < test_acc:
            self.max_acc = test_acc
            self.save_model()

        print("epoch {} valid acc {}".format(epoch + 1, test_acc / (batch_id + 1)))

    def run(self):
        self.load_model()
        self.load_data()

        for epoch in range(self.num_epochs):
            self.train_model(epoch)
            self.valid_model(epoch)

        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KoBERT model classifier fine tuning")
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
    parser.add_argument('--dropout_rate', default=0.5, type=float,
                        help="Dropout rate for classifier")
    parser.add_argument('--train_epochs', default=100, type=int,
                        help="Epochs for training")

    parser.add_argument('--save_path', default="save_model/best_model_past.pth", type=str,
                        help="save path for best accuracy model")
    parser.add_argument('--get_weights', default=False, type=bool,
                        help="get pretrained weights from saved model")

    args = parser.parse_args()
    print(args)

    solver = Solver(args)
    solver.run()
