import numpy as np
import random
import pandas as pd
import os
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import itertools
from IPython.display import display
import cntk as C
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set_style("darkgrid")

pd.set_option('display.max_rows', 20)
matplotlib.rcParams['figure.figsize'] = 20,7
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['axes.labelsize'] = 16

#
# Helper functions
#
def flatten(list):
    result = []
    for item in list:
        if (type(item) == type([])):
            items = flatten(item)
            result.extend(items)
        else:
            result.append(item)
    return result


def ColumnBox(items, columns):
    grid = []
    for i, item in enumerate(items):
        if (i<columns):
            grid.append([])
        grid[i%columns].append(item)

    return widgets.HBox(list(map(lambda x: widgets.VBox(x, layout=widgets.Layout(width=str(100//columns)+'%')), grid)))

#
# General functions used in multiple pages
#
def getNumDevicesToShow(df, deviceIDColumn):
    deviceCount = df[deviceIDColumn].unique()
    return widgets.IntSlider(
        value=10,
        min=1,
        max=len(deviceCount),
        step=1,
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='80%'))


#
# Acquire Data functions
#
def getDataSetChoices():
    files = []
    for entry in os.scandir('DataSets'):
        if entry.name.endswith('.csv') and entry.is_file():
            files.append(entry.name)

    defaultChoice = 'predictiveMaintenanceHydraulics.csv' if 'predictiveMaintenanceHydraulics.csv' in files else files[0]
    
    choices = widgets.Dropdown(
        options=files,
        description='Data Set:',
        value=defaultChoice,
        disabled=False)

    url = widgets.Text(
        value='',
        placeholder='Enter a URL to a csv file',
        description='URL:',
        disabled=False)

    name = widgets.Text(
        value='',
        placeholder='Enter a filename to save locally',
        description='Name:',
        disabled=False)

    return [choices, url, name]


def loadChoiceDataFrame(choices):
    if (choices[1].value != ''):
        df = pd.read_csv(choices[1].value)
        df.to_csv(os.path.join('DataSets', choices[2].value))
    else:
        df = pd.read_csv(os.path.join('DataSets', choices[0].value))

    return df


#
# Explore Data functions
#
def getDeviceIDColumnName(df):
    columnNames = df.columns.values.tolist()
    defaultDeviceID = 'machineID' if 'machineID' in columnNames else columnNames[0]
    return widgets.ToggleButtons(
        options=columnNames,
        value=defaultDeviceID,
        button_style='',
        tooltips=columnNames,
        disabled=False)

def getTimeColumnName(df):
    columnNames = df.columns.values.tolist()
    defaultTime = 'timeUntilFailure' if 'timeUntilFailure' in columnNames else columnNames[1]
    return widgets.ToggleButtons(
        options=columnNames,
        value=defaultTime,
        button_style='',
        tooltips=columnNames,
        disabled=False)

def getConfidenceLevel():
    return widgets.ToggleButtons(
        options=['80%','85%','90%','95%','99%'],
        value='95%',
        button_style='',
        disabled=False)

def getPopulationSize():
    return widgets.IntText(
        value=1000,
        disabled=False)

def getConfidenceLevelAndPopulation():
    confidenceLevelLabel = widgets.Label(
        value='Confidence Level:',
        layout=widgets.Layout(width='80%'))
    confidenceLevel = getConfidenceLevel()
    populationSizeLabel = widgets.Label(
        value='Desired Population Size:',
        layout=widgets.Layout(width='80%'))
    populationSize = getPopulationSize()
    ui = widgets.VBox([confidenceLevelLabel, confidenceLevel, populationSizeLabel, populationSize])
    return confidenceLevel, populationSize, ui

def calculateSampleSize(df, deviceIDColumn, p, N):
    s = len(df[deviceIDColumn].unique())
    zMap = {
        '80%': 1.28,
        '85%': 1.44,
        '90%': 1.65,
        '95%': 1.96,
        '99%': 2.58
    }
    z = zMap[p]
    p = 0.5
    c = 0.02
    cInv = 1/c
    sampleSize = ((z**2 * p * (1-p))/c**2)/(1+(((z**2 * p * (1-p))/c**2)-1)/(N))
    sampleSize = int(np.round(sampleSize))
    maximumPopulation = (s * (((cInv**2 * p**2 * z**2) - (cInv**2 * p * z**2)) + 1))/(((cInv**2 * p**2 * z**2) - (cInv**2 * p * z**2)) + s)
    maximumPopulation = int(np.round(maximumPopulation))
    
    return s, sampleSize, maximumPopulation

def getDeviceCountsConfiguration(df):
    deviceIDColumnLabel = widgets.Label(
        value='Device ID Column Name:',
        layout=widgets.Layout(width='80%'))
    deviceIDColumnName = getDeviceIDColumnName(df)
    timeColumnLabel = widgets.Label(
        value='Time Column Name:',
        layout=widgets.Layout(width='80%'))
    timeColumnName = getTimeColumnName(df)
    ui = widgets.VBox([deviceIDColumnLabel, deviceIDColumnName, timeColumnLabel, timeColumnName])
    return deviceIDColumnName, timeColumnName, ui


def displayDeviceCounts(df, deviceIDColumnName, timeColumnName):
    counts = np.array([])
    for i in df[deviceIDColumnName].unique():
        counts = np.append(counts, len(df.loc[df[deviceIDColumnName] == i]))

    unitNums = np.arange(0,len(df[deviceIDColumnName].unique()))
    
    plt.figure(1)
    plt.subplot(211)
    b1 = plt.bar(unitNums, counts)
    plt.xlabel('Device ID')
    plt.ylabel('Recorded Cycles')
    plt.show()

    plt.subplot(212)
    h1 = plt.hist(counts)
    plt.xlabel('Recorded Cycles')
    plt.ylabel('Devices')
    plt.show()


def displayFeatureHistograms(df, checkboxes):
    col_subplot = 4
    checked_boxes = [checkbox for checkbox in checkboxes if checkbox.value]
    total_checked = len(checked_boxes)
    row_subplot = int((total_checked+3) / col_subplot)
    fig, axarr = plt.subplots(row_subplot, col_subplot)
    figsize_width = matplotlib.rcParams.get('figure.figsize',[20,7])[0]
    figsize_height = 4*row_subplot
    fig.set_size_inches(figsize_width, figsize_height)
    for checkbox, ax in zip(checked_boxes, axarr.flatten()):
        h = ax.hist(df[checkbox.description])
        ax.set_xlabel(checkbox.description)
        ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()


def getFeaturesToShow(df, deviceIDColumn, timeColumn):
    columnNames = df.columns.values.tolist()
    columnNames.remove(deviceIDColumn)
    columnNames.remove(timeColumn)

    togglebuttons = []
    for col in columnNames:
        newToggleButton = widgets.ToggleButton(
            value=True,
            description=col,
            button_style='',
            tooltip=col,
            disable=False)
        togglebuttons.append(newToggleButton)

    ui = ColumnBox(togglebuttons, 6)

    return togglebuttons, ui


def getFeaturePlotsConfiguration(df, deviceIDColumn, timeColumn):
    labelCheckboxes = widgets.Label(
        value='Choose which features to show in a line plot:',
        layout=widgets.Layout(width='80%'))
    checkboxes, hbox = getFeaturesToShow(df, deviceIDColumn, timeColumn)

    labelDevices = widgets.Label(
        value='Number of Devices to Show:',
        layout=widgets.Layout(width='80%'))
    numDevices = getNumDevicesToShow(df, deviceIDColumn)

    ui = widgets.VBox([labelCheckboxes, hbox, labelDevices, numDevices])

    return checkboxes, numDevices, ui


def displayFeaturePlots(df, deviceColumnName, t, checkboxes, numDevices):
    deviceIDs = df[deviceColumnName].unique()
    for checkbox in checkboxes:
        if checkbox.value:
            for i in range(numDevices):
                x = df[t].loc[df[deviceColumnName] == deviceIDs[i]]
                y = df[checkbox.description].loc[df[deviceColumnName] == deviceIDs[i]]
                if (x.shape[0] > 500):
                    x = x.tail(500)
                    y = y.tail(500)
                x = x*-1
                plt.plot(x, y)
            plt.xlabel('Time')
            plt.ylabel(checkbox.description)
            plt.show()


#
# Prepare Data functions
#
def getWindowSize():
    return widgets.IntSlider(
        value=10,
        min=2,
        max=20,
        step=1,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='80%'))


def getFeaturesToSmooth(df):
    columnNames = df.columns.values.tolist()
    columnNames.remove('deviceID')
    columnNames.remove('time')

    togglebuttons = []
    for col in columnNames:
        newToggleButton = widgets.ToggleButton(
            value=True,
            description=col,
            button_style='',
            tooltip=col,
            disable=False)
        togglebuttons.append(newToggleButton)
    return togglebuttons


def getMovingAvgChosenFeaturesConfiguration(df):
    labelCheckboxes = widgets.Label(
        value='Choose which features to calculate the moving average:',
        layout=widgets.Layout(width='80%'))
    checkboxes = getFeaturesToSmooth(df)
    labelWindowSize = widgets.Label(
        value='Moving Average Window:',
        layout=widgets.Layout(width='80%'))
    windowSize = getWindowSize()

    ui = widgets.VBox([
        labelCheckboxes,
        ColumnBox(checkboxes, 6),
        labelWindowSize,
        windowSize])

    return checkboxes, windowSize, ui


def calculateMovingAvg(df, checkboxes, windowSize):
    dfFinal = pd.DataFrame()
    deviceIDs = df['deviceID'].unique()
    columnNames = df.columns.values

    for device in deviceIDs:
        dfSlice = df.loc[df['deviceID'] == device]
        dfSlice.columns = columnNames

        for checkbox in checkboxes:
            if checkbox.value:
                rollingMean = dfSlice[checkbox.description].rolling(windowSize, min_periods=windowSize).mean()
                meanName = checkbox.description + 'smoothed'
                dfSlice = pd.concat([dfSlice, rollingMean.rename(meanName)], axis=1)

        dfFinal = dfFinal.append(dfSlice, ignore_index=True)

    dfFinal.dropna(inplace=True)
    return dfFinal


def getNumDevicesToShowForMovingAvg(df):
    labelDevices = widgets.Label(
        value='Number of Devices to Show:',
        layout=widgets.Layout(width='80%'))
    numDevices = getNumDevicesToShow(df, 'deviceID')

    return numDevices, widgets.VBox([labelDevices, numDevices])


def displayMovingAvgFeatures(df, numDevices, checkboxes):
    deviceIDs = df['deviceID'].unique()
    checked_boxes = [checkbox for checkbox in checkboxes if checkbox.value]
    total_checked = len(checked_boxes)
    figsize_width = matplotlib.rcParams.get('figure.figsize',[20,7])[0]
    figsize_single_height = matplotlib.rcParams.get('figure.figsize',[20,7])[1]
    figsize_height = float(figsize_single_height) / 2 # * total_checked
    
    grouped = df.groupby('deviceID')
    for checkbox in checked_boxes:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(figsize_width, figsize_height)
        for i in range(numDevices):
            x = grouped.get_group(deviceIDs[i])['time']
            y = grouped.get_group(deviceIDs[i])[checkbox.description]
            if (x.shape[0] > 500):
                x = x.tail(500)
                y = y.tail(500)
            x = x*-1
            axs[0].plot(x, y)
        axs[0].set_xlabel('Time')
        axs[0].set_title(checkbox.description + ' Before Smoothing')
        ymin, ymax = axs[0].get_ylim()
        
        for i in range(numDevices):
            x = grouped.get_group(deviceIDs[i])['time']
            y = grouped.get_group(deviceIDs[i])[checkbox.description + 'smoothed']
            if (x.shape[0] > 500):
                x = x.tail(500)
                y = y.tail(500)
            x = x*-1
            axs[1].plot(x, y)
        axs[1].set_xlabel('Time')
        axs[1].set_title(checkbox.description + ' After Smoothing')
        axs[1].set_ylim([ymin,ymax])
        plt.tight_layout()
        plt.show()


def getFeaturesToKeep(df):
    columnNames = df.columns.values.tolist()
    columnNames.remove('deviceID')
    columnNames.remove('time')

    togglebuttons = []
    for col in columnNames:
        newToggleButton = widgets.ToggleButton(
            value=True,
            description=col,
            button_style='',
            tooltip=col,
            disable=False)
        togglebuttons.append(newToggleButton)
       
    labelCheckboxes = widgets.Label(
        value='Choose which features to keep:',
        layout=widgets.Layout(width='80%'))

    ui = widgets.VBox([
        labelCheckboxes,
        ColumnBox(togglebuttons, 6)])

    return togglebuttons, ui


def discardUnwantedFeatures(df, checkboxes):
    for checkbox in checkboxes:
        if not checkbox.value:
            df = df.drop(checkbox.description, axis=1)
    return df


def getTargetThreshold(df):
    maximum = df['time'].max()

    labelCheckboxes = widgets.Label(
        value='Threshold for Target:',
        layout=widgets.Layout(width='80%'))

    targetThreshold = widgets.IntSlider(
        value=10,
        min=1,
        max=240,
        step=1,
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='80%'))

    return targetThreshold, widgets.VBox([labelCheckboxes, targetThreshold])


def calculateTargetColumn(df, thresh):
    df['target'] = np.where(df['time'] <= thresh, 1, 0)
    df.drop(['time', 'deviceID'], axis=1, inplace=True)
    return df


def displayTargetHist(df):
    targ = [0,1]
    zeroes = len(df.loc[df['target'] == 0])
    ones = len(df.loc[df['target'] == 1])
    counts = [zeroes, ones]
    
    fig, ax = plt.subplots()
    ax.bar(targ, counts, facecolor='yellow', edgecolor='gray')
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[2] = 'No Maintenance Necessary'
    labels[6] = 'Ready for Maintenance'
    ax.set_xticklabels(labels)
    plt.show()


#
# Train Model functions
#

def getModelType():
    return widgets.ToggleButtons(
        options=['Deep Neural Net', 'Gradient Tree Booster', 'Random Forest', 'Support Vector Machine'],
        button_style='',
        description='Model:',
        disabled=False)


def getHyperparameters(modelName):
    if modelName == 'Random Forest':
        return {
            'n_estimators': random.randrange(500,2000,50),
            'min_samples_split': random.randrange(3,100),
            'min_samples_leaf': random.randrange(1,5),
            'random_state': random.randrange(1,999999999)
        }
    else:
        return {
            'random_state': random.randrange(1,999999999)
        }


#helper functions for CNTK implementation of DNN
def getMinibatch(features, labels, size):
    miniFeats = features[:size]
    miniLabels = labels[:size]
    newFeatures = features[size:]
    newLabels = labels[size:]
    return newFeatures, newLabels, miniFeats, miniLabels


def create_model(features):
    numHiddenLayers = random.randrange(2,3)
    hiddenLayersDim = random.randrange(40,55)
    numOutputClasses = 2
    with C.layers.default_options(init=C.layers.glorot_uniform(), activation=C.sigmoid):
        h = features
        for _ in range(numHiddenLayers):
            h = C.layers.Dense(hiddenLayersDim)(h)
        lastLayer = C.layers.Dense(numOutputClasses, activation = None)
        
        return lastLayer(h)

    
def trainDNN(trainX, trainY):
    numOutputClasses = 2
    
    newCol =  np.where(trainY == 0, 1, 0)
    newCol = pd.DataFrame(newCol)
    trainY = trainY.reset_index(drop=True)
    trainY = pd.concat([trainY, newCol], axis=1, ignore_index=True)
    inputDim = trainX.shape[1]
    trainX = np.ascontiguousarray(trainX.as_matrix().astype(np.float32))
    trainY = np.ascontiguousarray(trainY.as_matrix().astype(np.float32))
    
    input = C.input_variable(inputDim)
    label = C.input_variable(numOutputClasses)
    
    classifier = create_model(input)
    loss = C.cross_entropy_with_softmax(classifier, label)
    evalError = C.classification_error(classifier, label)

    learning_rate = 0.5
    lrSchedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch) 
    learner = C.sgd(classifier.parameters, lrSchedule)
    trainer = C.Trainer(classifier, (loss, evalError), [learner])

    minibatchSize = 25
    numSamples = trainX.shape[0] - (trainX.shape[0]%25)
    numMinibatchesToTrain = numSamples / minibatchSize

    #train the model
    for i in range(0, int(numMinibatchesToTrain)):
        trainX, trainY, features, labels = getMinibatch(trainX, trainY, minibatchSize)
        trainer.train_minibatch({input : features, label : labels})
    
    return [classifier,trainer,input,label]


def trainModel(modelName, trainX, trainY):
    hyperparams = getHyperparameters(modelName)

    if modelName == 'Support Vector Machine':
        classifier = SVC(random_state=hyperparams['random_state'])
    elif modelName == 'Deep Neural Net':
        classifier = trainDNN(trainX, trainY)
    elif modelName == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=hyperparams['n_estimators'],
                            min_samples_split=hyperparams['min_samples_split'],
                            min_samples_leaf=hyperparams['min_samples_leaf'],
                            random_state=hyperparams['random_state'],
                            n_jobs = 4)
    elif modelName == 'Gradient Tree Booster':
        classifier = GradientBoostingClassifier(random_state=hyperparams['random_state'])

    if modelName != 'Deep Neural Net':
        classifier.fit(trainX, trainY)
        
    return classifier


def shuffleAndNormalize(df):
    order = np.random.permutation(len(df))
    df = df.iloc[order]
    targs = df['target']
    df = (df - df.mean()) / (df.max() - df.min())
    df['target'] = targs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=1, inplace=True, how='all')
    df.dropna(axis=0, inplace=True, how='any')
    return df


def getTrainingSet(df):
    features = df.drop('target', axis=1)
    targets = df['target']
    splitIndex = (int)(df.shape[0]*.8)
    return features[:splitIndex], targets[:splitIndex]


def getTestingSet(df):
    features = df.drop('target', axis=1)
    targets = df['target']
    splitIndex = (int)(df.shape[0]*.8)
    return features[splitIndex:], targets[splitIndex:]


def getPredictions(model, testX, testY, modelName):
    if modelName == 'Deep Neural Net':
        #model = [classifier,trainer,input,label]
        classifier = model[0]
        trainer = model[1]
        input = model[2]
        label = model[3]
        
        newCol =  np.where(testY == 0, 1, 0)
        newCol = pd.DataFrame(newCol)
        testY = testY.reset_index(drop=True)
        testY = pd.concat([testY, newCol], axis=1, ignore_index=True)
        testY = np.ascontiguousarray(testY.as_matrix().astype(np.float32))
        testX = np.ascontiguousarray(testX.as_matrix().astype(np.float32))
        
        trainer.test_minibatch({input : testX, label : testY})
        out = C.softmax(classifier)
        
        predictedLabelProbs = out.eval({input : testX})
        predictedLabelProbs = pd.DataFrame(predictedLabelProbs)
        predictions = pd.DataFrame(np.where(predictedLabelProbs[predictedLabelProbs.columns[0]] >                                                                           predictedLabelProbs[predictedLabelProbs.columns[1]], 1, 0))
    else:
        predictions = model.predict(testX)
        
    return predictions

def getAccuracyMetrics(testY, predictions):
    accuracy = metrics.accuracy_score(testY, predictions)
    precision = metrics.precision_score(testY, predictions)
    recall = metrics.recall_score(testY, predictions)
    f1score = metrics.f1_score(testY, predictions)
    return accuracy, precision, recall, f1score

def plotConfusionMatrix(testY, predictions):
    plt.figure()
    cmReg = confusion_matrix(testY, predictions)
    cm = cmReg.astype('float') / cmReg.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    classes = ['No Maintenance Needed', 'Ready for Maintenance']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    plt.figure()
    cm = cmReg
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['No Maintenance Needed', 'Ready for Maintenance']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
