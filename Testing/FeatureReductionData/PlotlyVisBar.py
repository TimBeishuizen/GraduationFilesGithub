import numpy as np
import csv

import plotly.graph_objs as go
import plotly
import json

plotly.tools.set_credentials_file(username='TPABeishuizen', api_key='2A6fAPOKHrp4Nxl3Yhlx')

# data_name = 'MicroOrganisms'
# data_name = 'Arcene'
# data_name = 'RSCTC'
# data_name = 'Psoriasis'
data_name = 'average'

filter = True
wrapper = True
embedded = True

filter_values = []
wrapper_values = []
embedded_values = []

if filter:
    with open('New_Filter_values.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            filter_values.append(row)

    filter_values = np.asarray(filter_values)
else:
    filter_values = np.zeros((11, 0))

if wrapper:
    with open('New_Wrapper_values.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            wrapper_values.append(row)

    wrapper_values = np.asarray(wrapper_values)
else:
    wrapper_values = np.zeros((11, 0))

if embedded:
    with open('New_Embedded_values.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            embedded_values.append(row)

    embedded_values = np.asarray(embedded_values)
else:
    embedded_values = np.zeros((11, 0))

values = np.concatenate((filter_values, wrapper_values, embedded_values), axis=1)

plot_legend = []
val_score = []
test_score = []
n_feat = []
comp_time = []
precision = []
recall = []
F1_score = []
text_label = []
marker_type = []


if data_name == 'average':
    # Remove backward elimination
    values = values[:, values[1, :] != 'backward']

    # Find values per dataset
    MO_values = values[:, values[0, :] == 'MicroOrganisms']
    Arcene_values = values[:, values[0, :] == 'Arcene']
    RSCTC_values = values[:, values[0, :] == 'RSCTC']
    Psoriasis_values = values[:, values[0, :] == 'Psoriasis']

    # Initialize average value array
    avg_values = MO_values

    # Fill average value array
    for i in range(MO_values.shape[1]):
        avg_values[0, i] = 'average'
        avg_values[1:4, i] = MO_values[1:4, i]
        for j in range(5, 11):
            avg_values[j, i] = (float(MO_values[j, i]) + float(Arcene_values[j, i]) + float(RSCTC_values[j, i]) + float(Psoriasis_values[j, i])) / 4

    values = avg_values

# Keeping track of locations of algorithm types
forward = []
floating = []
PTA25 = []
PTA510 = []
embforwSVM = []
embforwRF = []
filterMI = []
filterT = []

# Plotting values
for i in range(values.shape[1]):
        if str(values[0, i]) != data_name:
            continue

        val_score.append(float(values[4, i]))
        test_score.append(float(values[5, i]))
        n_feat.append(float((values[6, i])))
        comp_time.append(float(values[7, i]))
        precision.append(float(values[8, i]))
        recall.append(float(values[9, i]))
        F1_score.append(float(values[10, i]))
        # type.append(str(values[1, i]))

        if values[1, i] in ['forward', 'backward', 'floating'] and values[3, i] in ['Random', 'MI']:
            text_label.append(values[1, i] + '<br> order: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            if values[1, i] == 'forward':
                marker_type.append(0)
                forward.append(i)
            elif values[1, i] == 'backward':
                marker_type.append(1)
            elif values[1, i] == 'floating':
                marker_type.append(2)
                floating.append(i)
        elif values[1, i][0:2] == '[2':
            text_label.append('PTA <br> [l, r] = ' + values[1, i] + '<br> order: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            marker_type.append(3)
            PTA25.append(i)
        elif values[1, i][0:2] == '[5':
            text_label.append('PTA <br> [l, r] = ' + values[1, i] + '<br> order: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            marker_type.append(4)
            PTA510.append(i)
        elif values[1, i] in ['embedded forward'] and values[3, i] in ['svm']:
            text_label.append(values[1, i] + '<br> ML: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            marker_type.append(13)
            embforwSVM.append(i)
        elif values[1, i] in ['embedded forward'] and values[3, i] in ['rf']:
            text_label.append(values[1, i] + '<br> ML: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            marker_type.append(17)
            embforwRF.append(i)
        elif values[1, i] == 'filter' and values[3, i] in ['MI']:
            text_label.append(values[1, i] + '<br> ranking method: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            marker_type.append(18)
            filterMI.append(i)
        elif values[1, i] == 'filter' and values[3, i] in ['T-test']:
            text_label.append(values[1, i] + '<br> ranking method: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            marker_type.append(22)
            filterT.append(i)
        else:
            print(values[1, i])
            raise ValueError

x = ["Filter: MI", "Filter: T-test", "Forward", "Floating", "PTA [2,5]", "PTA [5,10]", "Embedded: SVM", "Embedded: RF"]

trace1 = go.Bar(
    x=x,
    y=[comp_time[filterMI[0]], comp_time[filterT[0]], comp_time[forward[0]], comp_time[floating[0]], comp_time[PTA25[0]]
        , comp_time[PTA510[0]], comp_time[embforwSVM[0]], comp_time[embforwRF[0]]],
    text=[text_label[filterMI[0]], text_label[filterT[0]], text_label[forward[0]], text_label[floating[0]], text_label[PTA25[0]]
        , text_label[PTA510[0]], text_label[embforwSVM[0]], text_label[embforwRF[0]]],
    textposition = 'auto',
    marker=dict(
        color='rgb(200,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

trace2 = go.Bar(
    x=x,
    y=[comp_time[filterMI[1]], comp_time[filterT[1]], comp_time[forward[1]], comp_time[floating[1]], comp_time[PTA25[1]]
        , comp_time[PTA510[1]], comp_time[embforwSVM[1]], comp_time[embforwRF[1]]],
    text=[text_label[filterMI[1]], text_label[filterT[1]], text_label[forward[1]], text_label[floating[1]], text_label[PTA25[1]]
        , text_label[PTA510[1]], text_label[embforwSVM[1]], text_label[embforwRF[1]]],
    textposition = 'auto',
    marker=dict(
        color='rgb(160,201,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

trace3 = go.Bar(
    x=x,
    y=[comp_time[filterMI[2]], comp_time[filterT[2]], comp_time[forward[2]], comp_time[floating[2]], comp_time[PTA25[2]]
        , comp_time[PTA510[2]], comp_time[embforwSVM[2]], comp_time[embforwRF[2]]],
    text=[text_label[filterMI[2]], text_label[filterT[2]], text_label[forward[2]], text_label[floating[2]], text_label[PTA25[2]]
        , text_label[PTA510[2]], text_label[embforwSVM[2]], text_label[embforwRF[2]]],
    textposition = 'auto',
    marker=dict(
        color='rgb(120,201,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

trace4 = go.Bar(
    x=x[2:-2],
    y=[comp_time[forward[3]], comp_time[floating[3]], comp_time[PTA25[3]]
        , comp_time[PTA510[3]]],
    text=[text_label[forward[3]], text_label[floating[3]], text_label[PTA25[3]]
        , text_label[PTA510[3]]],
    textposition = 'auto',
    marker=dict(
        color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)


layout = go.Layout(
    title='Feature selection algorithm time performance',
    hovermode='closest',
    xaxis=dict(
        title='Algorithm type'),
    yaxis=dict(tickfont=dict(size=25),
               titlefont=dict(size=25),
        title='Computation time (s)'),
    legend=dict(orientation="h"),
    showlegend=False,
    font=dict(
        color="black",
        size=30
    ),
)

data = [trace1, trace2, trace3, trace4]

fig = dict(data=data, layout=layout)

plotly.plotly.plot(fig, filename='grouped-bar-direct-labels')