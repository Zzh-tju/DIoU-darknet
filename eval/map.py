import os
path = "/home/yourpath/evaluation/"     #Put all the mAP files (AP50, AP55,..., AP95) in this folder
mAPlist = []
for txt in os.listdir(path):
    content = open(path+ '/' + txt, 'r')
    line = content.readline()
    sum_AP = 0.0
    while line:
        sum_AP += float(line.split()[1])
        line = content.readline()
    mAP = sum_AP / 20
    mAPlist.append(mAP)
    content.close()
    content = open(path+ '/' + txt, 'a')
    content.write('\n' + '%f'%mAP)
    content.close()
print(sum(mAPlist)/len(mAPlist))
