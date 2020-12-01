import pandas as pd
import os
import argparse

from hangul_utils.hangul_utils import unicode
from filter import Filter


class Preprocessing(object):
    def __init__(self, config):
        self.data_path = config.load_csv_path
        self.save_path = config.save_txt_path
        self.data_type = config.abnormal_normal
        self.print_refined_data = config.print_data
        data_read_encoding = 'UTF-8'

        self.train = pd.read_csv(self.data_path, encoding=data_read_encoding)
        self.raw_value = [str(rv).split(',') for rv in self.train.values[:, 0]] # data column 추출

        self.dataSet = set()
        self.dataSet_refined = set()

        if config.remove_past_txt and os.path.exists(self.save_path):
            os.remove(self.save_path)

    def remove_duplicate(self):
        for rv in self.raw_value:
            try:
                self.dataSet.add(rv[0])
            except Exception as e:
                print(e)

    def print_dataset(self):
        for dt in self.dataSet_refined:
            print(dt, end='\n\n')

    def save_csv(self):
        with open(self.save_path, 'a') as f:
            f.write('id\tdata\tlabel\n')
            for i, dt in enumerate(self.dataSet_refined):
                f.write(str(i) + '\t')
                f.write(dt + '\t')
                f.write('0 \n')

    def run(self):
        filer = Filter()
        self.remove_duplicate()

        for dt in self.dataSet:
            dt = filer.phone_number_filter(dt)
            dt = filer.url_filter(dt)
            dt = filer.price_filter(dt)
            dt = filer.dm_filter(dt)
            dt = filer.special_char_filter(dt)

            dt = unicode.join_jamos(dt)

            self.dataSet_refined.add(dt)

        if self.print_refined_data:
            self.print_dataset()

        self.save_csv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Twitter Data Preprocessing")
    parser.add_argument('--abnormal_normal', type=int, help='Data type select(0: abnormal, 1: normal)')
    parser.add_argument('--load_csv_path', type=str, default='data/use_abnormal.csv', help='CSV file path to load')
    parser.add_argument('--save_txt_path', type=str, default='total.txt', help='TXT file path to save')
    parser.add_argument('--remove_past_txt', type=bool, default=False,
                        help='If True, remove past txt data and save new txt')
    parser.add_argument('--print_data', type=bool, default=False, help='If True, print refined dataset')
    args = parser.parse_args()

    pp = Preprocessing(args)
    pp.run()