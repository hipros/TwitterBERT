import gluonnlp
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class BERTDataset(Dataset):
    def __init__(self, dataset_path, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        # hearder 정보 버리기, 1번(content data),  2번(label) 추출
        dataset = gluonnlp.data.TSVDataset(dataset_path, num_discard_samples=3, field_indices=[1, 2])
        transform = gluonnlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx])for i in dataset]


    def __getitem__(self, ind):
        return (self.sentences[ind] + (self.labels[ind], ))

    def __len__(self):
        return (len(self.sentences))
