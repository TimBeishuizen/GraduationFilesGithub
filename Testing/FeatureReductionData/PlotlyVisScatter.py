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

FS_acc = []

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
else:
    values = values[:, values[0, :] == data_name]

# Plotting values
for i in range(values.shape[1]):
        if str(values[0, i]) != data_name:
            continue

        FS_acc.append(float(values[5, i]) * 0.99 ** float(values[6, i]))

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
            elif values[1, i] == 'backward':
                marker_type.append(1)
            elif values[1, i] == 'floating':
                marker_type.append(2)
        elif values[1, i][0:2] == '[2':
            text_label.append('PTA <br> [l, r] = ' + values[1, i] + '<br> order: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            marker_type.append(3)
        elif values[1, i][0:2] == '[5':
            text_label.append('PTA <br> [l, r] = ' + values[1, i] + '<br> order: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            marker_type.append(4)
        elif values[1, i] in ['embedded forward'] and values[3, i] in ['svm']:
            text_label.append(values[1, i] + '<br> ML: ' + values[3, i] + '<br> threshold: ' + values[6, i])
            marker_type.append(13)
        elif values[1, i] in ['embedded forward'] and values[3, i] in ['rf']:
            text_label.append(values[1, i] + '<br> ML: ' + values[3, i] + '<br> threshold: ' + values[6, i])
            marker_type.append(17)
        elif values[1, i] == 'filter' and values[3, i] in ['MI']:
            text_label.append(values[1, i] + '<br> ranking method: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            marker_type.append(18)
        elif values[1, i] == 'filter' and values[3, i] in ['T-test']:
            text_label.append(values[1, i] + '<br> ranking method: ' + values[3, i] + '<br> threshold: ' + values[2, i])
            marker_type.append(22)
        else:
            print(values[1, i])
            raise ValueError

trace_list = []

for i in range(len(n_feat)):
    trace_list.append(go.Scatter(
            x=[np.asarray(n_feat)[i]],
            y=[np.asarray(test_score)[i]],
            mode='markers',
            legendgroup=values[1, i],
            name=text_label[i],
            marker=dict(
                size=20,
                symbol=marker_type[i],
                color=comp_time[i],
                colorscale='Viridis',
                colorbar=dict(
                    title='Average computation time'
                ),
                showscale=False),
            text=text_label[i]
        ))

layout = go.Layout(
    title='Feature selection algorithm accuracy',
    hovermode='closest',
    xaxis=dict(tickfont=dict(size=15),
               titlefont=dict(size=15),
        title='Features preserved'),
    yaxis=dict(tickfont=dict(size=15),
               titlefont=dict(size=15),
        title='Test accuracy'),
    legend=dict(x=0,
                y=-0.2,
                orientation="h"),
    showlegend=False,
    font = dict(
    color="black",
    size=20)
)

updatemenus=list([
    dict(
        buttons=list([
            dict(
                args=[{'y': [test_score]}],
                label='Accuracy',
                method='restyle'
            ),
            dict(
                args=[{'y': [precision]}],
                label='Precision',
                method='restyle'
            ),
            dict(
                args=[{'y': [recall]}],
                label='Recall',
                method='restyle'
            ),
            dict(
                args=[{'y': [F1_score]}],
                label='F1',
                method='restyle'
            )
        ]),
        direction = 'left',
        pad = {'r': 10, 't': 10},
        showactive = True,
        type = 'buttons',
        x = 0.1,
        xanchor = 'left',
        y = 1.1,
        yanchor = 'top'
    ),
])

#layout['updatemenus'] = updatemenus

fig = dict(data=trace_list, layout=layout)
plotly.plotly.plot(fig, filename='tryout')