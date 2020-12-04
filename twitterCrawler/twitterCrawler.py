import GetOldTweets3 as got
import pandas as pd
import argparse


def main_hash_split(content):
    content_split = content.text.split('#')

    main_tweet = content_split[0]
    hash_tweet = [dt.replace('[\b*#]', '') for dt in content_split[1:]]

    return main_tweet, hash_tweet


class TwitterCrawler:
    def __init__(self, config):
        query_for_search = config.query_for_search
        number_for_search = config.max_tweets_num
        tweet_criteria = got.manager.TweetCriteria().setQuerySearch(query_for_search).setMaxTweets(number_for_search)

        self.hash_tag_set = set()
        self.content_text_set = set()
        self.tweet = got.manager.TweetManager.getTweets(tweet_criteria)
        self.print_content = config.print_content
        self.save_tsv_path = config.save_path

    def print_content_txt(self):
        for dt in self.content_text_set:
            print(dt)

    def save_content(self):
        twitter_df = pd.DataFrame(list(self.content_text_set),
                                  columns=['date', 'time', 'user_name', 'text', 'link', 'retweet_counts',
                                           'favorite_counts', 'user_created', 'user_tweets', 'user_followings',
                                           'user_followers'])

        twitter_df.to_csv(self.save_tsv_path, index=True, sep='\t')

    def run(self):
        for tw in self.tweet:
            main_content, hash_content = main_hash_split(tw)

            for ht in hash_content:
                if len(ht) < 8:
                    self.hash_tag_set.add(ht)
                else:
                    self.content_text_set.add(main_content)

            self.content_text_set.add(main_content)

        if self.print_content:
            self.print_content_txt()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Twitter Crawler from hash tags")
    parser.add_argument('--query_for_search', type=str, default="#조건 OR #ㅈㄱ", help='hash tag for search(#content1 OR #content2 OR ...)')
    parser.add_argument('--print_content', type=bool, default=False, help='If True, print contents')
    parser.add_argument('--max_tweets_num', type=int, default=10000, help='Maximum number of tweets for crawling')
    parser.add_argument('--save_path', type=str, default="sample.tsv", help='save file path')
    args = parser.parse_args()

    twc = TwitterCrawler(args)
    twc.run()
