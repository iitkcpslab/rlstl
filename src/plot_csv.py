import matplotlib.pyplot as plt
import csv

p = []

with open('log_bpw1.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            print(lines)     
            p.append(float(lines[2]))

plt.plot(p, label = "p",color="purple")

plt.savefig("bpw.png")
#plt.show()


