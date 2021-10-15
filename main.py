from BBC.BBC import main as BBC_main
from common import clear_prediction_results
from drugs.drugs import main as drugs_main

if __name__ == '__main__':
    print("\n\n---------------- QUESTION 1 ----------------")
    BBC_main()

    print("\n\n---------------- QUESTION 2 ----------------")
    clear_prediction_results()
    drugs_main()
