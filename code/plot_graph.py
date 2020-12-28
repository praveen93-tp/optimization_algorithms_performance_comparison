import os
import pandas as pd


import pandas as pd
import glob
import matplotlib.pyplot as plt

datapath = r"C:\Users\WINDOWS\Desktop\Results\CIFAR_CNN"
allfiles = glob.glob(datapath + "\*.xls")

"""
#loss graph
for excelfiles in allfiles:
    fname = excelfiles.split('\\')[-1].split('.')[0]
    raw_excel = pd.read_excel(excelfiles,fname)
    raw_excel.columns = ['Tloss', 'Vloss', 'TAcc', 'VAcc']
    df = raw_excel.iloc[1:52]
    plt.plot(df['Tloss'], 'g', label='Training loss')
    plt.plot(df['Vloss'], 'b', label='validation loss')
    plt.title(fname + ': ' + 'Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots_cfair_loss/' + fname + '_' + 'loss.png')
    plt.clf()

exit()
"""

#accuracy graphs

for excelfiles in allfiles:
    fname = excelfiles.split('\\')[-1].split('.')[0]
    raw_excel = pd.read_excel(excelfiles,fname)
    raw_excel.columns = ['Tloss', 'Vloss', 'TAcc', 'VAcc']
    df = raw_excel.iloc[1:52]
    plt.plot(df['TAcc'], 'g', label='Training Accuracy')
    plt.plot(df['VAcc'], 'b', label='Validation Accuracy')
    plt.title(fname + ': ' + 'Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('plots_cfair_accuracy/' + fname + '_' + 'acc.png')
    plt.clf()
exit()


t_loss = pd.DataFrame()
v_loss = pd.DataFrame()
t_acc = pd.DataFrame()
v_acc = pd.DataFrame()

for excelfiles in allfiles:
    fname = excelfiles.split('\\')[-1].split('.')[0]
    if fname=="adam" or fname=="Radam":
        raw_excel = pd.read_excel(excelfiles,fname)
        raw_excel.columns = ['Tloss', 'Vloss', 'TAcc', 'VAcc']
        df = raw_excel.iloc[1:]
        t_loss[fname]=df['Tloss']
        v_loss[fname] = df['Vloss']
        t_acc[fname] = df['TAcc']
        v_acc[fname] = df['VAcc']


#Training loss
col=['b','g','y','r','c','m','k']
number_of_cols=t_loss.columns
print(len(number_of_cols))
nm=number_of_cols[0].split('_')[0]
for index in range(len(number_of_cols)):
    plt.plot(t_loss[number_of_cols[index]], col[index])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(nm + ': ' + 'Trainingloss')
plt.show()
plt.clf()

for index in range(len(number_of_cols)):
    plt.plot(v_loss[number_of_cols[index]], col[index])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(nm + ': ' + 'Validationloss ')
plt.show()
plt.clf()


for index in range(len(number_of_cols)):
    plt.plot(t_acc[number_of_cols[index]], col[index])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(nm + ': ' + 'TrainingAccuracy')
plt.show()
plt.clf()

for index in range(len(number_of_cols)):
    plt.plot(v_acc[number_of_cols[index]], col[index])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(nm + ': ' + 'ValidationAccuracyComp')
plt.show()
plt.clf()


exit()