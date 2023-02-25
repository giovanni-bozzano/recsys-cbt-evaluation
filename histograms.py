import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'framework'))

import matplotlib.pyplot as plt

from framework.Data_manager.AmazonReviewData.AmazonMoviesTVReader import AmazonMoviesTVReader
from framework.Data_manager.Movielens.Movielens20MReader import Movielens20MReader
from framework.Data_manager.Movielens.MovielensHetrec2011Reader import MovielensHetrec2011Reader
from framework.Data_manager.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader


target_data_reader = AmazonMoviesTVReader()
dataset = target_data_reader.load_data()
dataset.get_URM_all().eliminate_zeros()
plt.hist(dataset.get_URM_all().data, bins=5)
plt.gca().set(xlabel='Rating', ylabel='Frequency')
plt.savefig('plots/amazon-movies-tv-series-ratings-frequency.png')
plt.close()

target_data_reader = MovielensHetrec2011Reader()
dataset = target_data_reader.load_data()
dataset.get_URM_all().eliminate_zeros()
plt.hist(dataset.get_URM_all().data, bins=10)
plt.gca().set(xlabel='Rating', ylabel='Frequency')
plt.savefig('plots/movielens-hetrec-2011-ratings-frequency.png')
plt.close()

target_data_reader = NetflixPrizeReader()
dataset = target_data_reader.load_data()
dataset.get_URM_all().eliminate_zeros()
plt.hist(dataset.get_URM_all().data, bins=5)
plt.gca().set(xlabel='Rating', ylabel='Frequency')
plt.savefig('plots/netflix-prize-ratings-frequency.png')
plt.close()

target_data_reader = Movielens20MReader()
dataset = target_data_reader.load_data()
dataset.get_URM_all().eliminate_zeros()
plt.hist(dataset.get_URM_all().data, bins=10)
plt.gca().set(xlabel='Rating', ylabel='Frequency')
plt.savefig('plots/movielens-20m-ratings-frequency.png')
plt.close()
