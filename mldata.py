import numpy as np
import math

def Build1DHistogramClassifier(X,T,B,xmin,xmax):
    HF = np.zeros(B).astype('int32')
    HM = np.zeros(B).astype('int32')
    binindices = ((np.round(((B-1)*(x-xmin)/(xmax-xmin)))).astype('int32') for x in X)
    for i,b in enumerate(binindices):
        if T[i] == 'Female':
            HF[b]+=1
        else:
            HM[b]+=1
    return [HF,HM]

B = 32
i = 0
X = []
T = []
file = open("meow.csv", "r")
data = file.read().splitlines()
for line in data:
    if i != 0:
        parts = line.split(",")
        feet = parts[0]
        inches = parts[1]
        X.append(int(feet)*12 + int(inches))
        T.append(parts[2])
    i = i + 1
xmin = min(X)
xmax = max(X)
print ("X min is: " + str(xmin))
print ("Xmax is " + str(xmax))

histograms = Build1DHistogramClassifier(X,T,B, xmin, xmax)

def BF(x):
    difference = xmax - xmin
    increment = difference/(B * 1.0)
    start = xmin
    bin = 0
    while True:

        if start > x:
            return bin - 1
        bin = bin + 1
        start = start + increment


HF = histograms[0]
HM = histograms[1]
print ("female histogram: " + str(HF))
print ("male histogram:" + str(HM))
print ("Histogram results")
print (HF[BF(55)]*1.0/(HF[BF(55)]+ HM[BF(55)]))
print (HF[BF(60)]*1.0/(HF[BF(60)]+ HM[BF(60)]))
print (HF[BF(65)]*1.0/(HF[BF(65)]+ HM[BF(65)]))
print (HF[BF(70)]*1.0/(HF[BF(70)]+ HM[BF(70)]))
print (HF[BF(75)]*1.0/(HF[BF(75)]+ HM[BF(75)]))
print (HF[BF(80)]*1.0/(HF[BF(80)] + HM[BF(80)]))


Xf = []
Xm = []
i = 0
for x in X:
    if T[i] == "Female":
        Xf.append(x)
    else:
        Xm.append(x)
    i = i + 1
def bayFormula(N,x,mean,stdev):

    return N*(1/(stdev*(2*math.pi)**.5) * math.e**(-.5*((x-mean)/stdev)**2))


Xfnum = len(Xf)
Xmnum = len(Xm)
Xfstdev = np.std(Xf)
Xmstdev = np.std(Xm)

Xfmean = np.mean(Xf)
Xmmean = np.mean(Xm)
print("Female mean: " + str(Xfmean))
print("Male mean: " + str(Xmmean))
print("Female stdev: " + str(Xfstdev))
print ("Male stdev: " + str(Xmstdev))
print ("Female sample size: " + str(Xfnum))
print ("Male sample size " + str(Xmnum))

print ("Bayseian results")
print (bayFormula(Xfnum,55,Xfmean,Xfstdev)/(bayFormula(Xfnum,55,Xfmean,Xfstdev) + bayFormula(Xmnum,55,Xmmean,Xmstdev)))
print (bayFormula(Xfnum,60,Xfmean,Xfstdev)/(bayFormula(Xfnum,60,Xfmean,Xfstdev) + bayFormula(Xmnum,60,Xmmean,Xmstdev)))
print (bayFormula(Xfnum,65,Xfmean,Xfstdev)/(bayFormula(Xfnum,65,Xfmean,Xfstdev) + bayFormula(Xmnum,65,Xmmean,Xmstdev)))
print (bayFormula(Xfnum,70,Xfmean,Xfstdev)/(bayFormula(Xfnum,70,Xfmean,Xfstdev) + bayFormula(Xmnum,70,Xmmean,Xmstdev)))
print (bayFormula(Xfnum,75,Xfmean,Xfstdev)/(bayFormula(Xfnum,75,Xfmean,Xfstdev) + bayFormula(Xmnum,75,Xmmean,Xmstdev)))
print (bayFormula(Xfnum,80,Xfmean,Xfstdev)/(bayFormula(Xfnum,80,Xfmean,Xfstdev) + bayFormula(Xmnum,80,Xmmean,Xmstdev)))


#
#####
##
print ("results for subset of data")

X = X[0:50]
T = T[0:50]

histograms = Build1DHistogramClassifier(X, T, B, xmin, xmax)

HF = histograms[0]
HM = histograms[1]
print ("male histogram: " + str(HM))
print ("female hisogram: " + str(HF))

print (HF[BF(55)] * 1.0 / (HF[BF(55)] + HM[BF(55)]))
#print HF[BF(60)] * 1.0 / (HF[BF(60)] + HM[BF(60)])
print (HF[BF(65)] * 1.0 / (HF[BF(65)] + HM[BF(65)]))
print (HF[BF(70)] * 1.0 / (HF[BF(70)] + HM[BF(70)]))
print (HF[BF(75)] * 1.0 / (HF[BF(75)] + HM[BF(75)]))
print (HF[BF(80)] * 1.0 / (HF[BF(80)] + HM[BF(80)]))

Xf = []
Xm = []
i = 0
for x in X:
    if T[i] == "Female":
        Xf.append(x)
    else:
        Xm.append(x)
    i = i + 1

Xfnum = len(Xf)
Xmnum = len(Xm)
Xfstdev = np.std(Xf)
Xmstdev = np.std(Xm)

Xfmean = np.mean(Xf)
Xmmean = np.mean(Xm)
print ("Female mean: " + str(Xfmean))
print ("Male mean: " + str(Xmmean))
print ("Female stdev: " +str(Xfstdev))
print ("Male stdev: " + str(Xmstdev))
print ("Female sample size: " + str(Xfnum))
print ("Male sample size: " + str(Xmnum))

print (bayFormula(Xfnum, 55, Xfmean, Xfstdev) / (bayFormula(Xfnum, 55, Xfmean, Xfstdev) + bayFormula(Xmnum, 55, Xmmean, Xmstdev)))
print (bayFormula(Xfnum, 60, Xfmean, Xfstdev) / (bayFormula(Xfnum, 60, Xfmean, Xfstdev) + bayFormula(Xmnum, 60, Xmmean, Xmstdev)))
print (bayFormula(Xfnum, 65, Xfmean, Xfstdev) / (bayFormula(Xfnum, 65, Xfmean, Xfstdev) + bayFormula(Xmnum, 65, Xmmean, Xmstdev)))
print (bayFormula(Xfnum, 70, Xfmean, Xfstdev) / (bayFormula(Xfnum, 70, Xfmean, Xfstdev) + bayFormula(Xmnum, 70, Xmmean, Xmstdev)))
print (bayFormula(Xfnum, 75, Xfmean, Xfstdev) / (bayFormula(Xfnum, 75, Xfmean, Xfstdev) + bayFormula(Xmnum, 75, Xmmean, Xmstdev)))
print (bayFormula(Xfnum, 80, Xfmean, Xfstdev) / (bayFormula(Xfnum, 80, Xfmean, Xfstdev) + bayFormula(Xmnum, 80, Xmmean, Xmstdev)))

B2 = np.alen(HF)
print (B2)