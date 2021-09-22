import pandas as panda

import questions.question_3 as question3

drugs_csv_model = panda.read_csv('drug200.csv')

# Question 3
question3.question3()

# Question 4

# def fetch_drugs_features_data(drugs_csv_model):
#     return drugs_csv_model.drop(DRUGS_FEATURES["DRUG"], axis=1).head()
#
#
# def fetch_drugs_target_data(drugs_csv_model):
#     return drugs_csv_model[DRUGS_FEATURES["DRUG"]].head()
#
#
# def get_naive_bayes_classifier(drugs_csv_model, features_vectors):
#     initial_naive_bayes_classifier = MultinomialNB(alpha=.01)
#     drugs_target_data = fetch_drugs_target_data(drugs_csv_model)
#     initial_naive_bayes_classifier.fit(features_vectors, drugs_target_data)
#     return initial_naive_bayes_classifier
#
#
# drugs_csv_model = panda.read_csv('drug200.csv')
# vectorizer = TfidfVectorizer()
#
# drugs_features = fetch_drugs_features_data(drugs_csv_model)
# features_vectors = vectorizer.fit_transform(drugs_features)
#
# naive_bayes_classifier = get_naive_bayes_classifier(drugs_csv_model, features_vectors)
#
# # testing
# print_prediction(naive_bayes_classifier, vectorizer)
