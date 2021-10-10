from drugs.questions.question_3 import question3
from drugs.questions.question_4_5_6 import question4_5_6
from drugs.questions.question_8 import question_8


def main():
    question3()
    file = open("drugs/outputs/drug-performance.txt", 'a+')
    question4_5_6(file)
    question_8(file)
