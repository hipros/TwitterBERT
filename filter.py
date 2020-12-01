import re


class Filter(object):
    def __init__(self):
        self.phone_number_pattern = [r'\d{2,3}[-\.\s]*\d{3,4}[-\.\s]*\d{4}(?!\d)',
                                     r'\(\d{3}\)\s*\d{4}[-\.\s]??\d{4}']
        self.url_pattern = [r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),|]|(?:%[0-9a-fA-F][0-9a-fA-F]))+']
        self.price_pattern = [r'\d{1,3}[,\.]\d{1,3}[만\천]?\s?[원]|\d{1,5}[만\천]?\s?[원]',
                              r'[일/이/삼/사/오/육/칠/팔/구/십/백][만\천]\s?[원]',
                              r'(?!-)\d{2,4}[0]{2,4}(?!년)(?!.)|\d{1,3}[,/.]\d{3}']
        self.dm_pattern = [r'@\w*\s']
        self.special_char_pattern = [r'\\', r'[^\w\s]', r'"']

        self.phone_number_replace_str = 'tel'
        self.url_replace_str = 'url'
        self.price_replace_str = 'money'
        self.dm_replace_str = 'dm'
        self.special_char_replace_str = ''

    def phone_number_filter(self, text):
        for re_pattern in self.phone_number_pattern:
            text = re.sub(re_pattern, self.phone_number_replace_str, text)

        return text

    def url_filter(self, text):
        for re_pattern in self.url_pattern:
            text = re.sub(re_pattern, self.url_replace_str, text)

        return text

    def price_filter(self, text):
        for re_pattern in self.price_pattern:
            text = re.sub(re_pattern, self.price_replace_str, text)

        return text

    def dm_filter(self, text):
        for re_pattern in self.dm_pattern:
            text = re.sub(re_pattern, self.dm_replace_str, text)

        return text

    def special_char_filter(self, text):
        for re_pattern in self.special_char_pattern:
            text = re.sub(re_pattern, self.special_char_replace_str, text)

        return text
