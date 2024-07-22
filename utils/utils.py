import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product

def plotItem(dataframe, category, name, type, rotate = False, all_provided = pd.DataFrame()):
    item = sorted(dataframe[category].unique().tolist())
    if all_provided.__len__() != 0:
        item = sorted(all_provided[category].unique().tolist())
    item_count = []

    for i in item:
        counter = int(dataframe[dataframe[category] == i][category].count())
        item_count.append(counter)

    if type == 'pie':
        plt.pie(
            item_count,
            labels=item,
            autopct='%1.1f%%',
        )
        plt.title(f"{category} in {name} Dataset")
        plt.show()
    
    if type == 'bar':
        x_coords = np.arange(len(item_count))
        plt.bar(x_coords, item_count, tick_label=item)
        if rotate == True:
            plt.xticks(rotation=90)
        plt.ylabel('Samples')
        plt.title(f'{category} for {name} Dataset')
        plt.show()

def fixLabels(dataframe, labels, column):
  for broken, fixed in labels.items():
    dataframe.loc[dataframe[column] == broken, column] = fixed
  print(f"The labels are {dataframe[column].unique()}")

def ratiosDictionary(labels, dataframe, category):
    count = []
    for i in labels:
        counter = int(dataframe[dataframe[category] == i][category].count())
        count.append(counter)
    uniqueness_ratios = []
    total_count = sum(count)
    for i in count:
        uniqueness_ratios.append(i/total_count)
    return {labels[idx]:uniqueness_ratios[idx] for idx in range(len(labels))}

def WeightUniquenessScoreCalculate(x, weight, ratios_dictionary):
    return (1-ratios_dictionary[x])*weight

def UniquenessScoreGenerate(dataframe, category_list, weight_list):
    if len(category_list) == len(weight_list):
        new_dataframe_labels = []
        for idx in range(len(category_list)):
            labels = sorted(dataframe[category_list[idx]].unique().tolist())
            ratios_dictionary = ratiosDictionary(labels, dataframe, category_list[idx])
            dataframe[f'{category_list[idx]}_US'] = dataframe[category_list[idx]].apply(lambda x: WeightUniquenessScoreCalculate(x, weight_list[idx], ratios_dictionary))
            new_dataframe_labels.append(f'{category_list[idx]}_US')
        dataframe['Total_US'] = dataframe[new_dataframe_labels].sum(axis=1)
        return dataframe.sort_values(by=['Total_US'], ascending=False)
    else:
        print("there needs to be one weight for every category, please try again")

def ParetoScoreGenerate(dataframe):
    uniqueness_list = dataframe['Total_US'].to_list()
    uniqueness_list_sum = sum(uniqueness_list)
    dataframe['pareto_proportion'] = (dataframe['Total_US']/uniqueness_list_sum)*100
    functional_list = dataframe['pareto_proportion'].to_list()
    uniqueness_pareto = []
    grab = 0
    for i in functional_list:
        grab += i
        uniqueness_pareto.append(grab)
    dataframe['pareto_distribution'] = uniqueness_pareto
    return dataframe

def TestTrainScore(test_proportion, dataframe):
    pareto_list = dataframe['pareto_distribution'].to_list()
    idx = int(len(pareto_list)*(1-test_proportion))
    return pareto_list[idx]

def GridSearchWeights(dataframe, category_list, weight_matrix, test_proportion, plot = False):
    weight_matrix_update = []
    for i in weight_matrix:
        grab_list = []
        for j in range(int((i[1]-i[0])/i[2])+1):
            grab = "{:.2f}".format(i[0]+(j*i[2]))
            grab_list.append(float(grab))
        weight_matrix_update.append(grab_list)

    complete_set = []
    for items in product(*weight_matrix_update):
        complete_set.append([list(items), 0])

    weight_matrix_correction = []
    for i in complete_set:
        if sum(i[0]) == 100:
            weight_matrix_correction.append(i)
    complete_set = weight_matrix_correction

    max_idx = 0
    max_score = 0
    for idx in range(len(complete_set)):
        dataframe = UniquenessScoreGenerate(dataframe, category_list, complete_set[idx][0])
        dataframe = ParetoScoreGenerate(dataframe)
        complete_set[idx][1] = TestTrainScore(test_proportion, dataframe)
        if complete_set[idx][1] > max_score:
            max_score = complete_set[idx][1]
            max_idx = idx
    
    if plot == True:
        score_list = []
        for i in complete_set:
            score_list.append(i[1])
        iter_list = [i for i in range(len(score_list))]

        plt.plot(
        iter_list,
        score_list
        )
        plt.title("Evaluating Bias")
        plt.ylabel("Percentage of data covered by training set")
        plt.xlabel("Test")
        plt.show()

    return complete_set[max_idx], complete_set


    