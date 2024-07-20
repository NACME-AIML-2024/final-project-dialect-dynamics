import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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