import pandas as panda

from common import plot_instances, set_cwd


def question3():
    set_cwd()
    drugs_csv_model = panda.read_csv('drugs/drug200.csv')
    drug_column_counts = drugs_csv_model.Drug.value_counts()

    drugA_count = drug_column_counts.drugA
    drugB_count = drug_column_counts.drugB
    drugC_count = drug_column_counts.drugC
    drugX_count = drug_column_counts.drugX
    drugY_count = drug_column_counts.drugY

    drug_names = sorted(drugs_csv_model['Drug'].unique())
    drug_values = [drugA_count, drugB_count, drugC_count, drugX_count, drugY_count]
    plot_instances('drug-distribution.pdf', drug_names, drug_values)
