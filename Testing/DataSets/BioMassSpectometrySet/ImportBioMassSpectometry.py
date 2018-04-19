import urllib.request
import numpy as np
import os
import csv

url_bio = r'https://www.openml.org/data/get_csv/1593707/phpHyLSNF'

response = urllib.request.urlopen(url_bio)
data = response.read()      # a `bytes` object
text = data.decode('utf-8') # a `str`; this step can't be used if data is binary

rows = text.split('\n')
rows.remove('')

instances = []

for row in rows:
    instance = row.split(',')
    instances.append(instance)

for i in range(len(instances[0])):
    instances[0][i] = instances[0][i].replace('"', '')

with open('bio_mass_spect_data.csv', 'w', newline='') as csv_file:
    csvwriter = csv.writer(csv_file)
    csvwriter.writerows(instances)
