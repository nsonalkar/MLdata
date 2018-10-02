import numpy as np
import numpy.linalg as la
import math
N = 0
i = 0
H = []
G = []
S = []

file = open("data2.csv", "r")
data = file.read().splitlines()
for line in data:
    if i != 0:
        parts = line.split(",")
        feet = parts[0]
        inches = parts[1]
        H.append(float(parts[1]))
        G.append(parts[0])
        S.append(float(parts[2]))
        N = N + 1
    i = i + 1

N = N / 2
B = math.log(N,2) + 1
B = int(B) + 1
B= 10
H = np.array(H)
G = np.array(G)
S = np.array(S)

H_min = min(H)
H_max = max(H)
S_min = min(S)
S_max = max(S)

HF = np.zeros((B,B))
HM = np.zeros((B,B))

def rowMap(x):
    return (B-1)* ((x-H_min)/(H_max - H_min))

def colMap(x):
    return (B-1) * ((x - S_min)/(S_max - S_min))

for x in range(0, len(H)):
    H_i = H[x]
    S_i = S[x]
    G_i = G[x]
    row = rowMap(H_i)
    col = colMap(S_i)
    if G_i == "Female":
        HF[row][col] += 1
    else:
        HM[row][col] += 1
print B
print HF[rowMap(69)][colMap(17.5)]/(HF[rowMap(69)][colMap(17.5)] + HM[rowMap(69)][colMap(17.5)])
print HF[rowMap(66)][colMap(22)]/(HF[rowMap(66)][colMap(22)] + HM[rowMap(66)][colMap(22)])
print HF[rowMap(70)][colMap(21.5)]/(HF[rowMap(70)][colMap(21.5)] + HM[rowMap(70)][colMap(21.5)])
print HF[rowMap(69)][colMap(23.5)]/(HF[rowMap(69)][colMap(23.5)] + HM[rowMap(69)][colMap(23.5)])
# HF[row][col]/(HF[row][col] + HM[row][col])
# if B is a decimal number do we round up or down.

HFem = []
HMal = []
SFem = []
SMal = []
NMal = 0
NFem = 0

for x in range(0, len(H)):
    if G[x] == "Female":
        HFem.append(H[x])
        SFem.append(S[x])
        NFem += 1
    else:
        HMal.append(H[x])
        SMal.append(S[x])
        NMal += 1
HFem = np.array(HFem)
HMal = np.array(HMal)
SFem = np.array(SFem)
SMal = np.array(SMal)

Fem_mean = np.mean([HFem,SFem], axis = 1)
Mal_mean = np.mean([HMal,SMal], axis = 1)

Fem_cov = np.cov([HFem, SFem])
Mal_cov = np.cov([HMal, SMal])




def pdfMale(x):
    powerMal = -.5*np.dot(np.dot((x-Mal_mean),la.inv(Mal_cov)),np.transpose(x-Mal_mean))
    powerFem = -.5 * np.dot(np.dot((x - Fem_mean), la.inv(Fem_cov)), np.transpose(x - Fem_mean))
    topPart = NMal*(1/(2*math.pi*math.pow(la.det(Mal_cov), .5)))*math.pow(math.e, powerMal)
    bottomPart = NFem*(1/(2*math.pi*math.pow(la.det(Fem_cov), .5 )))*math.pow(math.e, powerFem) + topPart
    return topPart/bottomPart


def pdfFemale(x):
    powerFem = -.5 * np.dot(np.dot((x - Fem_mean), la.inv(Fem_cov)), np.transpose(x - Fem_mean))
    powerMal = -.5 * np.dot(np.dot((x - Mal_mean), la.inv(Mal_cov)), np.transpose(x - Mal_mean))
    topPart = NFem * (1 / (2 * math.pi * math.pow(la.det(Fem_cov), .5))) * math.pow(math.e, powerFem)
    bottomPart = NMal * (1 / (2 * math.pi * math.pow(la.det(Mal_cov), .5))) * math.pow(math.e, powerMal) + topPart
    return topPart / bottomPart

print pdfFemale([69,17.5])
print pdfFemale([66,22])
print pdfFemale([70,21.5])
print pdfFemale([69,23.5])


print HF
print HM

print NFem
print NMal



