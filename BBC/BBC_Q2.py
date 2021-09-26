#something wrong with import here
from common import plot_instances
import os

# Question 2

business_files_count = len(os.listdir("BBC\\business"))
entertainment_files_count = len(os.listdir("BBC\\entertainment"))
politics_files_count = len(os.listdir("BBC\\politics"))
sport_files_count = len(os.listdir("BBC\\sport"))
tech_files_count = len(os.listdir("BBC\\tech"))

news_names = ['business', 'entertainment', 'politics', 'sport', 'tech']
news_values = [business_files_count, entertainment_files_count, politics_files_count, sport_files_count, tech_files_count]
news_pdf = 'BBC-distribution.pdf'
plot_instances(news_pdf, news_names, news_values)
