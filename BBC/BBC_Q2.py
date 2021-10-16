from common import plot_instances, set_cwd
import os


# Question 2

def question_2():
    set_cwd()
    business_files_count = len(os.listdir("BBC\\BBC\\business"))
    entertainment_files_count = len(os.listdir("BBC\\BBC\\entertainment"))
    politics_files_count = len(os.listdir("BBC\\BBC\\politics"))
    sport_files_count = len(os.listdir("BBC\\BBC\\sport"))
    tech_files_count = len(os.listdir("BBC\\BBC\\tech"))

    news_names = ['business', 'entertainment', 'politics', 'sport', 'tech']
    news_values = [business_files_count, entertainment_files_count, politics_files_count, sport_files_count,
                   tech_files_count]
    news_pdf = 'BBC/outputs/BBC-distribution.pdf'
    plot_instances(news_pdf, news_names, news_values)
