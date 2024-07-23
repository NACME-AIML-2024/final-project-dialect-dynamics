import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import re

def plotItem(dataframe, category, name, type, rotate = False, all_provided = pd.DataFrame(), colors = ()):
    item = sorted(dataframe[category].unique().tolist())
    if all_provided.__len__() != 0:
        item = sorted(all_provided[category].unique().tolist())
    item_count = []

    for i in item:
        counter = int(dataframe[dataframe[category] == i][category].count())
        item_count.append(counter)

    if type == 'pie':
        if len(colors) == len(item):
            plt.pie(
                item_count,
                labels=item,
                autopct='%1.1f%%',
                colors=colors
            )
        else:
            plt.pie(
                item_count,
                labels=item,
                autopct='%1.1f%%'
            )
        plt.title(f"{category} in {name} Dataset")
        plt.show()
    
    if type == 'bar':
        x_coords = np.arange(len(item_count))
        if len(colors) == len(item):
            plt.bar(
                x_coords, 
                item_count, 
                tick_label=item,
                color=list(colors),
                )
        else:
            plt.bar(
                x_coords, 
                item_count, 
                tick_label=item
                )
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

def CategoricalUniquenessScoreGenerate(dataframe, category_list, weight_list):
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

def GridSearchWeights(dataframe, category_list, weight_matrix, test_proportion, plot = False, colors = [], title = 0):
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
    min_score = 10000000
    for idx in range(len(complete_set)):
        dataframe = CategoricalUniquenessScoreGenerate(dataframe, category_list, complete_set[idx][0])
        dataframe = ParetoScoreGenerate(dataframe)
        complete_set[idx][1] = TestTrainScore(test_proportion, dataframe)
        if complete_set[idx][1] > max_score:
            max_score = complete_set[idx][1]
            max_idx = idx
        if complete_set[idx][1] < min_score:
            min_score = complete_set[idx][1]
    
    if plot == True:
        score_list = []
        for i in complete_set:
            score_list.append(i[1])
        iter_list = [i for i in range(len(score_list))]

        total_range = max_score - min_score
        percentile_80 = min_score + 0.8*total_range
        percentile_60 = min_score + 0.6*total_range
        percentile_40 = min_score + 0.4*total_range
        percentile_20 = min_score + 0.2*total_range
        top_80 = [val>percentile_80 for val in score_list]
        top_60 = [val>percentile_60 and val<percentile_80 for val in score_list]
        top_40 = [val>percentile_40 and val<percentile_60 for val in score_list]
        top_20 = [val>percentile_20 and val<percentile_40 for val in score_list]
        bottom_20 = [val<percentile_20 for val in score_list]
        if len(colors) == 5:
            plt.scatter([xv for xv, ygt in zip(iter_list, bottom_20) if ygt],
                        [yv for yv, ygt in zip(score_list, bottom_20) if ygt], color=colors[0])
            plt.scatter([xv for xv, ygt in zip(iter_list, top_20) if ygt],
                        [yv for yv, ygt in zip(score_list, top_20) if ygt], color=colors[1]) 
            plt.scatter([xv for xv, ygt in zip(iter_list, top_40) if ygt],
                        [yv for yv, ygt in zip(score_list, top_40) if ygt], color=colors[2]) 
            plt.scatter([xv for xv, ygt in zip(iter_list, top_60) if ygt],
                        [yv for yv, ygt in zip(score_list, top_60) if ygt], color=colors[3]) 
            plt.scatter([xv for xv, ygt in zip(iter_list, top_80) if ygt],
                        [yv for yv, ygt in zip(score_list, top_80) if ygt], color=colors[4]) 
        else:
            plt.scatter([xv for xv, ygt in zip(iter_list, bottom_20) if ygt],
                        [yv for yv, ygt in zip(score_list, bottom_20) if ygt], color='blue')
            plt.scatter([xv for xv, ygt in zip(iter_list, top_20) if ygt],
                        [yv for yv, ygt in zip(score_list, top_20) if ygt], color='green') 
            plt.scatter([xv for xv, ygt in zip(iter_list, top_40) if ygt],
                        [yv for yv, ygt in zip(score_list, top_40) if ygt], color='orange') 
            plt.scatter([xv for xv, ygt in zip(iter_list, top_60) if ygt],
                        [yv for yv, ygt in zip(score_list, top_60) if ygt], color='yellow') 
            plt.scatter([xv for xv, ygt in zip(iter_list, top_80) if ygt],
                        [yv for yv, ygt in zip(score_list, top_80) if ygt], color='red')
        if title != 0:
            plt.title(title)
        else:
            plt.title("Evaluating Bias")
        plt.ylabel("Percentage of data covered by training set")
        plt.xlabel("Test")
        plt.show()

    return complete_set[max_idx], complete_set

def RecursiveGridSearchWeights(dataframe, category_list, weight_matrix, test_proportion, plot = False):
    best_set, complete_set = GridSearchWeights(dataframe, 
                                               category_list, 
                                               weight_matrix, 
                                               test_proportion, 
                                               plot, 
                                               colors = ['#409CFF', '#7D7AFF', '#BF5AF2', '#8944AB', '#FF375F'],
                                               title = 'First Pass: Resolution of 5%')
    for i in range(len(weight_matrix)):
        weight_matrix[i][0] = best_set[0][i]-4
        weight_matrix[i][1] = best_set[0][i]+5
        weight_matrix[i][2] = 1    
    best_set, complete_set = GridSearchWeights(dataframe, 
                                               category_list, 
                                               weight_matrix, 
                                               test_proportion, 
                                               plot, 
                                               colors = ['#DA8FFF', '#8944AB', '#FF6482', '#FF375F', '#D30F45'],
                                               title = 'Second Pass: Resolution of 1%')
    for i in range(len(weight_matrix)):
        weight_matrix[i][0] = best_set[0][i]-0.5
        weight_matrix[i][1] = best_set[0][i]+0.5
        weight_matrix[i][2] = 0.1  
    best_set, complete_set = GridSearchWeights(dataframe, 
                                               category_list, 
                                               weight_matrix, 
                                               test_proportion, 
                                               plot, 
                                               colors = ['#FF6482', '#FF6961', '#FF375F', '#D30F45', '#FF0015'],
                                               title = 'Third Pass: Resolution of 0.1%')
    return best_set, complete_set

def clean_content(content):
    # Remove anything within [ … ], / … /, < … >, and ( … )
    content = re.sub(r'\[.*?\]|\<.*?\>|\/.*?\/|\(.*?\)', '', content)
    # Remove special symbols while keeping punctuation marks
    content = re.sub(r'[^a-zA-Z0-9\s]', '', content)
    # Capitalize all text
    content = content.upper()
    # Strip leading and trailing whitespaces
    content = content.strip()
    return content

def WordsUniquenessScoreGenerate(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].apply(clean_content)
    dataframe = dataframe[dataframe[column_name] != '']

    def grabWords(string):
        return string.split()

    dataframe['Transcript_Words'] = dataframe[column_name].apply(lambda x: grabWords(x))
    spoken_matrix = dataframe['Transcript_Words'].to_list()

    all_spoken_words = []
    for i in spoken_matrix:
        for j in range(len(i)):
            all_spoken_words.append(i[j])

    def unique(list1):
        unique_dictionary = {}
        for x in list1:
            if x not in unique_dictionary:
                unique_dictionary.update({x : 1})
            if x in unique_dictionary:
                unique_dictionary[x] += 1
        return unique_dictionary

    spoken_unique_words = unique(all_spoken_words)

    sorted_spoken_unique_words = dict(sorted(spoken_unique_words.items(), key=lambda item: item[1], reverse=True))

    amount_of_words = len(all_spoken_words)

    def phraseUniqueScore(sorted_spoken_unique_words, all_spoken_words, amount_of_words):
        list_of_scores = []
        for i in all_spoken_words:
            list_of_scores.append(sorted_spoken_unique_words[i] / amount_of_words)
        return sum(list_of_scores)

    dataframe['Words_US'] = dataframe['Transcript_Words'].apply(lambda x: phraseUniqueScore(sorted_spoken_unique_words, x, amount_of_words))
    dataframe = dataframe.sort_values(by=['Words_US'], ascending=False)
    return dataframe

