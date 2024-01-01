import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

def generate_random_numbers(minimum, maximum, mean, std_dev):
    return np.clip(np.random.normal(mean, std_dev, 10**7), minimum, maximum)


def find_best_fit_distribution_continuous(data, distributions=None):
    """
    Find the best-fit distribution for the given data.

    Parameters:
        data (array-like): The data to fit the distributions to.
        distributions (list, optional): The candidate distributions to consider.
            Defaults to [stats.norm, stats.gamma, stats.expon, stats.uniform].

    Returns:
        best_distribution (scipy.stats.rv_continuous): The best-fit distribution.
        best_params (tuple): The parameters of the best-fit distribution.
    """
    if distributions is None:
        distributions = [
            scipy.stats.norm,      # Normal distribution
            scipy.stats.gamma,     # Gamma distribution
            scipy.stats.expon,     # Exponential distribution
            scipy.stats.uniform    # Uniform distribution
        ]

    best_distribution = None
    best_params = {}
    best_sse = np.inf

    for distribution in distributions:
        params = distribution.fit(data)
        sse = np.sum((distribution.pdf(data, *params) - data) ** 2)

        if sse < best_sse:
            best_distribution = distribution
            best_params = params
            best_sse = sse

    return best_distribution, best_params


PRODBRU = []
PRODBRUexZERO = []
PRODSTO = []
PRODSTOexZERO = []

# BRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUB
print("BRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRUBRU")

# lees de data van BRU in
directory = R'..\DATA\INPUT\daily_production\BRU'
dataBRU = []
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        with open(os.path.join(directory, filename)) as f:
            json_data = json.load(f)
            dataBRU.append(json_data)
dfBRU = pd.DataFrame(dataBRU)

# sla excel op met data van BRU
dfBRU.to_excel(r'..\DATA\INTERMEDIATE\DFBRU.xlsx', index=False)

# de dagen dat er gepland onderhoud was, daar houden we geen rekening mee
dfBRU = dfBRU.loc[dfBRU.maintenance != 'Yes', :]
dfBRU.to_excel(r'..\DATA\INTERMEDIATE\DFBRUexMAINTENANCE.xlsx', index=False)

# verzamel de productiecijfers BRU in een list
for x in range(0, len(dfBRU), 1):
    PRODBRU.append(dfBRU.iloc[x, 7])

# toon grafiek van de data in dfBRU
plt.plot(dfBRU["date"], dfBRU["production"])
plt.title("PRODUCTIE BRU ex onderhoud")
plt.xlabel("Datum")
plt.ylabel("Productie")
plt.show()
plt.hist(PRODBRU, bins=100)
plt.title("PRODUCTIE BRU ex onderhoud")
plt.xlabel("Volume")
plt.show()

# bepaal de verdeling
print("BRU")
print("deze berekeningen houden GEEN rekening met geplande onderhouden")
print("deze berekeningen houden WEL rekening met dagen dat er 0 productie was")
best_distribution, best_params = find_best_fit_distribution_continuous(PRODBRU)
print(f"Best-fit distribution: {best_distribution.name}")
print(f"Parameters: {best_params}")

# de dagen dat er 0 productie was, daar houden we geen rekening mee
dfBRU = dfBRU.loc[dfBRU.production != 0, :]
dfBRU.to_excel(r'..\DATA\INTERMEDIATE\DFBRUexMAINTENANCEexZERO.xlsx', index=False)

# verzamel de productiecijfers BRU in een list
for x in range(0, len(dfBRU), 1):
    PRODBRUexZERO.append(dfBRU.iloc[x, 7])

# toon grafiek van de data in dfBRU
plt.plot(dfBRU["date"], dfBRU["production"])
plt.title("PRODUCTIE BRU ex onderhoud ex 0")
plt.xlabel("Datum")
plt.ylabel("Productie")
plt.show()
plt.hist(PRODBRUexZERO, bins=100)
plt.title("PRODUCTIE BRU ex onderhoud ex 0")
plt.xlabel("Volume")
plt.show()

print("BRU")
print("deze berekeningen houden GEEN rekening met geplande onderhouden")
print("deze berekeningen houden GEEN rekening met dagen dat er 0 productie was")
best_distribution, best_params = find_best_fit_distribution_continuous(PRODBRUexZERO)
print(f"Best-fit distribution: {best_distribution.name}")
print(f"Parameters: {best_params}")

# simulatie maken
dagen = int(input("Hoeveel dagen simuleren ??? "))
for x in range(dagen):
    y = generate_random_numbers(np.min(PRODBRUexZERO), np.max(PRODBRUexZERO), np.mean(PRODBRUexZERO),
                                np.std(PRODBRUexZERO))
    print(f"DAG {x} : {int(np.mean(y))}")

# STOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOS
print("STOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTOSTO")

# lees de data van STO in
directory = R'..\DATA\INPUT\daily_production\STO'
dataSTO = []
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        with open(os.path.join(directory, filename)) as f:
            json_data = json.load(f)
            dataSTO.append(json_data)
dfSTO = pd.DataFrame(dataSTO)
dfSTO.to_excel(r'..\DATA\INTERMEDIATE\DFSTO.xlsx', index=False)

# de dagen dat er gepland onderhoud was, daar houden we geen rekening mee
dfSTO = dfSTO.loc[dfSTO.maintenance != 'Yes', :]
dfBRU.to_excel(r'..\DATA\INTERMEDIATE\DFSTOexMAINTENANCE.xlsx', index=False)

# verzamel de productiecijfers STO in een list
for x in range(0, len(dfSTO), 1):
    PRODSTO.append(dfSTO.iloc[x, 7])

# toon grafiek van de data in dfSTO
plt.plot(dfSTO["date"], dfSTO["production"])
plt.title("PRODUCTIE STO ex onderhoud")
plt.xlabel("Datum")
plt.ylabel("Productie")
plt.show()
plt.hist(PRODSTO, bins=100)
plt.title("PRODUCTIE STO ex onderhoud")
plt.xlabel("Volume")
plt.show()

# bepaal de verdeling
print("STO")
print("deze berekeningen houden GEEN rekening met geplande onderhouden")
print("deze berekeningen houden WEL rekening met dagen dat er 0 productie was")
best_distribution, best_params = find_best_fit_distribution_continuous(PRODSTO)
print(f"Best-fit distribution: {best_distribution.name}")
print(f"Parameters: {best_params}")

# de dagen dat er gepland onderhoud was, daar houden we geen rekening mee
dfSTO = dfSTO.loc[dfSTO.production != 0, :]
dfBRU.to_excel(r'..\DATA\INTERMEDIATE\DFSTOexMAINTENANCEexZERO.xlsx', index=False)

# verzamel de productiecijfers STO in een list
for x in range(0, len(dfSTO), 1):
    PRODSTOexZERO.append(dfSTO.iloc[x, 7])

# toon grafiek van de data in dfBRU
plt.plot(dfSTO["date"], dfSTO["production"])
plt.title("PRODUCTIE STO ex onderhoud ex 0")
plt.xlabel("Datum")
plt.ylabel("Productie")
plt.show()
plt.hist(PRODSTOexZERO, bins=100)
plt.title("PRODUCTIE STO ex onderhoud ex 0")
plt.xlabel("Volume")
plt.show()

# bepaal de verdeling
print("STO")
print("deze berekeningen houden GEEN rekening met geplande onderhouden")
print("deze berekeningen houden GEEN rekening met dagen dat er 0 productie was")
best_distribution, best_params = find_best_fit_distribution_continuous(PRODSTOexZERO)
print(f"Best-fit distribution: {best_distribution.name}")
print(f"Parameters: {best_params}")

# simulatie maken
dagen = int(input("Hoeveel dagen simuleren ??? "))
for x in range(dagen):
    y =generate_random_numbers(np.min(PRODSTOexZERO), np.max(PRODSTOexZERO), np.mean(PRODSTOexZERO),
                               np.std(PRODSTOexZERO))
    print(f"DAG {x} : {int(np.mean(y))}")