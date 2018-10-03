import numpy as np
import csv

import plotly.graph_objs as go
import plotly
import json

plotly.tools.set_credentials_file(username='TPABeishuizen', api_key='2A6fAPOKHrp4Nxl3Yhlx')

class_results = []

data_name = 'Hepatitis'

with open('MissingValueHandlingClassificationValues.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        class_results.append(row)

values = np.asarray(class_results)

plot_legend = []
val_score = []
test_score = []
comp_time = []
precision = []
recall = []
F1_score = []
text_label = []
marker_type = []

if data_name == 'Average':
    # Remove backward elimination

    # Find values per dataset
    Hepatitis_values = values[:, values[0, :] == 'Hepatitis']
    Cirrhosis_values = values[:, values[0, :] == 'Cirrhosis']
    Cervical_values = values[:, values[0, :] == 'Cervical']

    # Initialize average value array
    avg_values = Hepatitis_values

    # Fill average value array
    for i in range(Hepatitis_values.shape[1]):
        avg_values[0, i] = 'Average'
        avg_values[1:3, i] = Hepatitis_values[1:3, i]
        for j in range(4, 9):
            avg_values[j, i] = (float(Hepatitis_values[j, i]) + float(Cirrhosis_values[j, i]) +
                                float(Cervical_values[j, i])) / 3

    values = avg_values

else:
    values = values[:, values[0, :] == data_name]

# Plotting values
for i in range(values.shape[1]):
    if str(values[0, i]) != data_name:
        continue

    val_score.append(float(values[3, i]))
    test_score.append(float(values[4, i]))
    comp_time.append(float(values[5, i]))
    precision.append(float(values[6, i]))
    recall.append(float(values[7, i]))
    F1_score.append(float(values[8, i]))

    text_label.append(values[1, i] + ': ' + values[2, i])
    if values[1, i] == 'LD':
        marker_type.append(0)
    elif values[1, i] == 'SI':
        marker_type.append(1)
    elif values[1, i] == 'MI':
        marker_type.append(2)
    else:
        raise ValueError("%s is not a known missing value handling type" % values[1, i])

trace_list = []

for i in range(len(comp_time)):

    trace_list.append(go.Scatter(
        x=[np.log10(np.asarray(comp_time)[i])],
        y=[np.asarray(F1_score)[i]],
        mode='markers+text',
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
        text=text_label[i],
        textfont=dict(size=20),
        textposition='top right'
    ))

layout = go.Layout(
    title="%s %s " % (data_name, 'missing value handling algorithm F1 score,<br> after removing features with more than 15% missing values'),
    hovermode='closest',
    xaxis=dict(tickfont=dict(size=20),
               titlefont=dict(size=20),
               title='Computation time, logarithmic scale (s)'),
    yaxis=dict(tickfont=dict(size=20),
               titlefont=dict(size=20),
               title='Accuracy ([0, 1])'),
               #range=[0.50, 0.8]),
    legend=dict(x=0,
                y=-0.1,
                orientation="h"),
    showlegend=True,
    font=dict(
        color="black",
        size=20)
)

updatemenus = list([
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
        direction='left',
        pad={'r': 10, 't': 10},
        showactive=True,
        type='buttons',
        x=0.1,
        xanchor='left',
        y=1.1,
        yanchor='top'
    ),
])

# layout['updatemenus'] = updatemenus

fig = dict(data=trace_list, layout=layout)
plotly.plotly.plot(fig, filename='tryout')