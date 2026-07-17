from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import LeaveOneOut, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy.stats import norm
import statistics
import numpy as np
import pandas as pd
import os
import csv
import itertools
from IPython.display import display


def createHistogramStats(X, y, stdDevMult, saveOutput, outputDirectory):
    # 

    # Normalize the data and build a table
    # -- axis=0 processes each column independently
    numRows, numFeatures = X.shape
    X =  X.iloc[:,0:numFeatures]
    scaledArray = normalize(X, axis=0)
    scaledData = pd.DataFrame(scaledArray, columns=X.columns)
    fullData = pd.concat([y, scaledData], axis=1)    
    indicesTable = pd.DataFrame(range(0, numRows), columns=['indices'])
    fullData = pd.concat([indicesTable,fullData], axis=1)

    # Sort based on y (ie Diagnosis)
    sortedData = fullData.sort_values(by=['Diagnosis','indices'], ascending=[True, True],ignore_index=True)
    indicesB = sortedData.index[sortedData["Diagnosis"] == "B"].tolist()
    indicesM = sortedData.index[sortedData["Diagnosis"] == "M"].tolist()
    dataFinal = sortedData.drop(columns=['indices','Diagnosis'])
    dataFinal = dataFinal.iloc[:,range(0,numFeatures)]
    dataImageB = dataFinal.iloc[indicesB,:]
    dataImageM = dataFinal.iloc[indicesM,:]

    # Create stats for each column
    meanB = dataImageB.mean()
    stdDevB = dataImageB.std()
    trueMinB = dataImageB.min()
    trueMaxB = dataImageB.max()
    #
    meanM = dataImageM.mean()
    stdDevM = dataImageM.std()
    trueMinM = dataImageM.min()
    trueMaxM = dataImageM.max()
    #
    minB = meanB - stdDevMult*stdDevB
    maxB = meanB + stdDevMult*stdDevB
    minM = meanM - stdDevMult*stdDevM
    maxM = meanM + stdDevMult*stdDevM

    # Create stats tables to return
    # -- featureNames.to_frame(index=False) creates a new DataFrame with one column per entry in featureNames
    featureNames = dataImageB.columns
    labelTable = featureNames.to_frame(index=False)
    colNames = ['feature','mean', 'StdDev', 'min', 'max', 'true min', 'true max']
    # -- B table
    array_2d = np.column_stack((meanB, stdDevB, minB, maxB, trueMinB, trueMaxB))
    dataTable = pd.DataFrame(array_2d)
    statsTableB = pd.concat([labelTable,dataTable], axis=1)
    statsTableB.columns = colNames
    # -- M table
    array_2d = np.column_stack((meanM, stdDevM, minM, maxM, trueMinM, trueMaxM))
    dataTable = pd.DataFrame(array_2d)
    statsTableM = pd.concat([labelTable,dataTable], axis=1)
    statsTableM.columns = colNames
    
    # Process all features to find overlaps in the two distributions
    overlapCt = []
    minCol = 3
    maxCol = 4
    for i in range(numFeatures):   
        if (statsTableM.iloc[i,minCol] > statsTableB.iloc[i,minCol]):
            minThreshold = statsTableM.iloc[i,minCol]
            maxThreshold = statsTableB.iloc[i,maxCol]
            filtered_df = dataFinal[(dataFinal.iloc[:,i]>minThreshold) & (dataFinal.iloc[:,i]<maxThreshold)]
        else:
            minThreshold = statsTableB.iloc[i,minCol]
            maxThreshold = statsTableM.iloc[i,maxCol]
            filtered_df = dataFinal[(dataFinal.iloc[:,i]>minThreshold) & (dataFinal.iloc[:,i]<maxThreshold)]            
        numOverlaps = len(filtered_df)                
        overlapCt.append(numOverlaps)        
    overlapTable = pd.DataFrame(overlapCt)

     # Process all features to find all outliers   
    outliersM_ct = []
    outliersB_ct = []
    for i in range(numFeatures):
        # B table
        minThreshold = statsTableB.iloc[i,minCol]               
        maxThreshold = statsTableB.iloc[i,maxCol]  
        filtered_MinOutliers = dataImageB[(dataImageB.iloc[:,i]<=minThreshold)]
        filtered_MaxOutliers = dataImageB[(dataImageB.iloc[:,i]>=maxThreshold)]
        numOutliers = len(filtered_MinOutliers) + len(filtered_MaxOutliers) 
        outliersB_ct.append(numOutliers)
        
        # M table
        minThreshold = statsTableM.iloc[i,minCol]               
        maxThreshold = statsTableM.iloc[i,maxCol]  
        filtered_MinOutliers = dataImageM[(dataImageM.iloc[:,i]<=minThreshold)]
        filtered_MaxOutliers = dataImageM[(dataImageM.iloc[:,i]>=maxThreshold)]
        numOutliers = len(filtered_MinOutliers) + len(filtered_MaxOutliers) 
        outliersM_ct.append(numOutliers)
    outliersTableB = pd.DataFrame(outliersB_ct)
    outliersTableM = pd.DataFrame(outliersM_ct)
    extraStatsTable = pd.concat([labelTable, overlapTable, outliersTableB, outliersTableM], axis=1)
    colNames = ['feature','Overlaps','OutliersB','OutliersM']
    extraStatsTable.columns = colNames

    if (saveOutput==1):
        fullFileName = os.path.join(outputDirectory, 'HistogramStats.xlsx')
        extraStatsTable.to_excel(fullFileName)
    
    return fullData, sortedData, dataFinal, statsTableM, statsTableB, extraStatsTable

def engineeredFeatures(inputData, y, feature_list, columnNames, fusionMethodFlag):
    
    # Initialized the new DataFrame for the engineered features
    newX = pd.DataFrame()

    # fusionMethodFlag
    # -- 0 = mult
    # -- 1 = add
    
    # Do the engineering
    numLists = len(feature_list)
    for i in range(numLists):
        curList = feature_list[i]
        curFeatureCt = len(curList)
        curFeatureIndex = curList[0]
        curFeature = inputData.iloc[:,curFeatureIndex]
        for j in range(1, curFeatureCt):
            curFeatureIndex = curList[j]
            if (fusionMethodFlag == 0):
                curFeature = curFeature*inputData.iloc[:,curFeatureIndex]    
            else:
                curFeature = curFeature + inputData.iloc[:,curFeatureIndex]    
        newX.insert(i, columnNames[i], curFeature) 
    
    # Create the new dataframe
    scaledArray = normalize(newX, axis=0)
    scaledData = pd.DataFrame(scaledArray, columns=newX.columns)
    fullData = pd.concat([y,scaledData], axis=1)
    
    return newX, fullData

def removeSelectedRows(X, y, removalList):
    # Create a new table from X with an indices column
    numRows, numCols = X.shape    
    indicesTable = pd.DataFrame(range(0, numRows), columns=['indices'])    
    newData = pd.concat([indicesTable,X], axis=1)
    newY = pd.concat([indicesTable,y], axis=1)
    
    numRemovalRows = len(removalList)
    for i in range(numRemovalRows):
        curRow = removalList[i]
        newData.drop(newData[newData['indices'] == curRow].index, inplace=True)
        newY.drop(newY[newY['indices']==curRow].index, inplace=True)
        
    newData.reset_index(drop=True, inplace=True)
    X = newData.drop(columns=['indices'])
    newY.reset_index(drop=True, inplace=True)
    y = newY.drop(columns=['indices'])

    return X, y
       
def breastCancer_createBoxPlot1(fullData):
    # Reshape the data
    cols = fullData.columns
    melted_df = fullData.melt(id_vars="Diagnosis", value_vars=cols)
    
    # Plot all features at once
    plt.figure(figsize=(10,6))
    sns.boxplot(data=melted_df, x="value", y="variable", hue="Diagnosis")
    plt.show()
    
def breastCancer_createBoxPlot(X, y, numFeatures, saveImages, outputDirectory):
   
    # Normalize the data and build a table
    X =  X.iloc[:,0:numFeatures]
    scaledArray = normalize(X, axis=0)
    scaledData = pd.DataFrame(scaledArray, columns=X.columns)
    fullData = pd.concat([y, scaledData], axis=1)
    
    # Reshape the data
    # -- the melt format is needed for seaborn plots
    cols = fullData.columns
    melted_df = fullData.melt(id_vars="Diagnosis", value_vars=cols)
    
    # Plot all features at once
    plt.figure(figsize=(10,6))
    sns.boxplot(data=melted_df, x="value", y="variable", hue="Diagnosis")
    if (saveImages==1):
        fullFileName = os.path.join(outputDirectory, 'BoxPlotImage.jpg')
        plt.savefig(fullFileName,dpi=600, bbox_inches='tight')
    plt.show()

def breastCancer_createPlot4(inputRowsDataTable, subjectLabels, plotLabel, saveImages, normFlag, outputDirectory, plotLegend, fileNameTitle, offset):
    # Create a plot of stacked spectra

    
    # Initializations
    spectralTable = inputRowsDataTable
    [numRows, numCols] = spectralTable.shape
    xAxisValues = [x+1 for x in range(numCols)]
    spectralArray = spectralTable.to_numpy()
    
    # Plot the inputRowsTable traces
    for i in range(numRows):
        yTrace = spectralArray[i,:]
        if (normFlag==1):
            yTrace = normalize(yTrace[:, np.newaxis], axis=0)
        yTraceOffset = yTrace + (i*offset)
        tempLabel = f"Subject {subjectLabels[i]}"
        plt.plot(xAxisValues, yTraceOffset, label=tempLabel, linewidth=2)
                
    # Label the plot
    plt.title(plotLabel)
    plt.xlabel('Features')
    plt.ylabel('Offset Traces')
    if (plotLegend==1):
        plt.legend(fontsize=8,framealpha=0.3)
    if (saveImages==1):
        fullFileName = os.path.join(outputDirectory, fileNameTitle)
        plt.savefig(fullFileName,dpi=600, bbox_inches='tight')        
    plt.show()    
    
def breastCancer_createPlot3(goodRowsDataTable, plotLabel, avgClass2Label, saveImages, normFlag, outputDirectory, plotLegend, fileNameTitle):    
    # Find and plot the average of the goodRowsTable traces
    # -- doesn't seem to be called currently (6/16/26)

    # Initializations
    spectralTable = goodRowsDataTable
    [numRows, numCols] = spectralTable.shape
    xAxisValues = [x+1 for x in range(numCols)]
    goodSpectralArray = spectralTable.to_numpy()
    yTrace = np.mean(goodSpectralArray, axis=0)
    if (normFlag==1):        
        yTrace = normalize(yTrace[:, np.newaxis], axis=0)
    plt.plot(xAxisValues, yTrace, label=avgClass2Label, linewidth=3)

    # Label the plot
    plt.title(plotLabel)
    plt.xlabel('Features')
    plt.ylabel('Scores')
    if (plotLegend==1):
        plt.legend(fontsize=8)
    if (saveImages==1):
        fullFileName = os.path.join(outputDirectory, fileNameTitle)
        plt.savefig(fullFileName,dpi=600, bbox_inches='tight')        
    plt.show()    

def breastCancer_createPlot2Helper(xAxisValues, yTrace1, yTrace2, label1, label2, plotTitle, outputDirectory, plotFileName, saveImages):

    # Setup the plot
    fig, ax = plt.subplots()
    lw = 2  
    ax.plot(xAxisValues, yTrace1, label=label1, linewidth=lw)
    ax.plot(xAxisValues, yTrace2, label=label2, linewidth=lw)

    # Show the plot
    ax.set_title(plotTitle)
    ax.set_xlabel('Features')
    ax.set_ylabel('Scores')
    ax.legend(fontsize=8)
    if (saveImages==1):
        fullFileName = os.path.join(outputDirectory, plotFileName)
        plt.savefig(fullFileName,dpi=600, bbox_inches='tight')        
    plt.show() 
    
def breastCancer_createPlot2(X, y, sorted_df, saveImages, normFlag, outputDirectory, titlePrefix):
    # -- need error traps for the unlikely cases that there are no FPs, FNs, TPs, or TNs
    # -- note that 'Subjects' in sorted_df are now 1s-based indices into the original data table (updated on 6/16/26)
    # -- X and y have the same number of entries as sorted_df (can be less entries than the original data table)

    # 7/4/26
    # -- normalize each column independently
    scaledArray = normalize(X, axis=0)
    scaledData = pd.DataFrame(scaledArray, columns=X.columns)   

    # scale the data
    scaler = StandardScaler()
    scaledArray = scaler.fit_transform(scaledData)
    scaledData = pd.DataFrame(scaledArray, columns=X.columns) 
    
    # Initializations
    [numRows, numCols] = scaledData.shape
    xAxisValues = [x+1 for x in range(numCols)]

    # FP spectra
    # -- added check on 7/9/26
    avgFPLabel = 'Avg FP Feature Vector'
    error2Table = sorted_df[sorted_df['Error2'] == 1]
    numSpectra = len(error2Table)
    if (numSpectra==0):
        yTrace1 = []
    else:
        error2Indices = error2Table['Indices']
        error2Data = scaledData.iloc[error2Indices,:]
        error2SpectralArray = error2Data.to_numpy()
        yTrace1 = np.mean(error2SpectralArray, axis=0)
        if (normFlag==1):
            yTrace1 = normalize(yTrace1[:, np.newaxis], axis=0)
    
    # FN spectra
    # -- added check on 7/9/26
    avgFNLabel = 'Avg FN Feature Vector'
    error1Table = sorted_df[sorted_df['Error1'] == 1]
    numSpectra = len(error1Table)
    if (numSpectra==0):
        yTrace2 = []
    else:
        error1Indices = error1Table['Indices']
        error1Data = scaledData.iloc[error1Indices,:]
        error1SpectralArray = error1Data.to_numpy()
        yTrace2 = np.mean(error1SpectralArray, axis=0)
        if (normFlag==1):
            yTrace2 = normalize(yTrace2[:, np.newaxis], axis=0)

    # TP spectra
    # -- added check on 7/9/26
    avgTPLabel = 'Avg TP Feature Vector'
    tpTable1 = sorted_df[sorted_df['GT'] == 1]
    tpTable = tpTable1[tpTable1['Error']==0]
    numSpectra = len(tpTable)
    if (numSpectra==0):
        yTrace3 = []
    else:
        tpIndices = tpTable['Indices']
        tpData = scaledData.iloc[tpIndices,:]
        tpSpectralArray = tpData.to_numpy()
        yTrace3 = np.mean(tpSpectralArray, axis=0)
        if (normFlag==1):
            yTrace3 = normalize(yTrace3[:, np.newaxis], axis=0)
    
    # TN spectra
    # -- -- added check on 7/9/26
    avgTNLabel = 'Avg TN Feature Vector'
    tnTable1 = sorted_df[sorted_df['GT'] == 0]
    tnTable = tnTable1[tnTable1['Error']==0]    
    numSpectra = len(tnTable)
    if (numSpectra==0):
        yTrace4 = []
    else:
        tnIndices = tnTable['Indices']
        tnData = scaledData.iloc[tnIndices,:]
        tnSpectralArray = tnData.to_numpy()
        yTrace4 = np.mean(tnSpectralArray, axis=0)
        if (normFlag==1):
            yTrace4 = normalize(yTrace4[:, np.newaxis], axis=0)

    # Plot the first two spectra
    if len(yTrace1) != 0 and len(yTrace2) != 0:
        plotTitle = 'Class Feature Traces'
        plotFileName = titlePrefix + '_FeatureAvgTracePlot1.jpg'
        breastCancer_createPlot2Helper(xAxisValues, yTrace1, yTrace2, avgFPLabel, avgFNLabel, plotTitle, outputDirectory, plotFileName, saveImages)

    # Plot the second two spectra
    if len(yTrace3) != 0 and len(yTrace4) != 0:
        plotTitle = 'Class Feature Traces'
        plotFileName = titlePrefix + '_FeatureAvgTracePlot2.jpg'    
        breastCancer_createPlot2Helper(xAxisValues, yTrace3, yTrace4, avgTPLabel, avgTNLabel, plotTitle, outputDirectory, plotFileName, saveImages)    

    # Plot the FP and the TN
    if len(yTrace1) != 0 and len(yTrace4) != 0:
        plotTitle = 'Class Feature Traces'
        plotFileName = titlePrefix + '_FeatureAvgTracePlot3.jpg'    
        breastCancer_createPlot2Helper(xAxisValues, yTrace1, yTrace4, avgFPLabel, avgTNLabel, plotTitle, outputDirectory, plotFileName, saveImages)    

    # Plot the FN and the TP
    if len(yTrace2) != 0 and len(yTrace3) != 0:
        plotTitle = 'Class Feature Traces'
        plotFileName = titlePrefix + '_FeatureAvgTracePlot4.jpg'    
        breastCancer_createPlot2Helper(xAxisValues, yTrace2, yTrace3, avgFNLabel, avgTPLabel, plotTitle, outputDirectory, plotFileName, saveImages)        

    # Save trace values
    if saveImages==1 and len(yTrace1)!=0 and len(yTrace2)!=0 and len(yTrace3)!=0 and len(yTrace4)!=0:
        df = pd.DataFrame({'Indices':xAxisValues, 'FPs':yTrace1.flatten(), 'FNs':yTrace2.flatten(), 'TP':yTrace3.flatten(), 'TN':yTrace4.flatten()})
        tempTitle = titlePrefix + '_yTraceValues.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        df.to_excel(fullFileName) 
    
    return scaledData, yTrace1, yTrace2, yTrace3, yTrace4
    
def breastCancer_createPlot1(badRowsDataTable, goodRowsDataTable, plotLabel, avgClass2Label, showOutlierTracesFlag, saveImages, normFlag, \
                             outputDirectory, plotLegend, fileNameTitle):

    # Initializations and plot badRowsTable spectra
    # -- NOTE that normalizing every trace will put the outliers on the same scale as the inlier average trace
    # -- badRowsDataTable and goodRowsDataTable have two prepended columns (full indices, diagnosis)
    spectralTable = badRowsDataTable.iloc[:,2:]
    [numRows, numCols] = spectralTable.shape
    xAxisValues = [x+1 for x in range(numCols)]
    rowLabels = spectralTable.index.tolist() 
    rowLabels2 = badRowsDataTable.iloc[:,0]
    rowLabels2 = [x + 1 for x in rowLabels2]  # The index is 0's-based
    spectralArray = spectralTable.to_numpy()
    if showOutlierTracesFlag==1:
        for i in range(numRows):
            yTrace = spectralArray[i,:]
            if (normFlag==1):
                yTrace = normalize(yTrace[:, np.newaxis], axis=0)
            tempLabel = f"Subject {rowLabels2[i]} ({rowLabels[i]})"
            plt.plot(xAxisValues, yTrace, label=tempLabel)
    else:
        yTrace = np.mean(spectralArray, axis=0)
        if (normFlag==1):        
            yTrace = normalize(yTrace[:, np.newaxis], axis=0)
        plt.plot(xAxisValues, yTrace, label='Avg Outliers')    
        
    # Add the average of the goodRowsTable traces
    spectralTable = goodRowsDataTable.iloc[:,2:]
    spectralArray = spectralTable.to_numpy()
    yTrace = np.mean(spectralArray, axis=0)
    if (normFlag==1):        
        yTrace = normalize(yTrace[:, np.newaxis], axis=0)
    plt.plot(xAxisValues, yTrace, label=avgClass2Label, linewidth=3)

    # Label the plot
    plt.title(plotLabel)
    plt.xlabel('Features')
    plt.ylabel('Scores')
    if (plotLegend==1):
        plt.legend(fontsize=8)
    if (saveImages==1):
        fullFileName = os.path.join(outputDirectory, fileNameTitle)
        plt.savefig(fullFileName,dpi=600, bbox_inches='tight')        
    plt.show()    

def breastCancer_processOutliers1(resultImage, sortedData, numFeatures, stdDevThresholdMult):
    # The following ...
    # -- resultImage was formed from a correlation image of subject vectors of features -- could be for "B" or "M"
    # -- sortedData is sorted based on "B" than "M" subjects and has a index column into the original data table 
    #    -- 1st column is "indices" and 2nd column is "Diagnosis" and remaining numFeatures columns have the feature data
    # -- find the subjects resulting in very low scores (ie, < mean - 2.5*StdDev)

    # Get the stats
    row_sums = resultImage.sum(axis=1)
    tempMean = statistics.mean(row_sums)
    tempStdDev = statistics.stdev(row_sums)
    threshold = tempMean - stdDevThresholdMult*tempStdDev

    # Find the "bad" rows
    indices = [i for i, val in enumerate(row_sums) if val < threshold]
    badRowsTable = sortedData.iloc[indices,:]
    col1 = badRowsTable.iloc[:,0].reset_index(drop=True)
    col1 = pd.DataFrame(col1)
    col2 = badRowsTable.iloc[:,1].reset_index(drop=True)
    col2 = pd.DataFrame(col2)
    col3 = pd.DataFrame(indices, columns=['Image Indices']).reset_index(drop=True)
    col4 = row_sums[indices]
    col4 = pd.DataFrame(col4).reset_index(drop=True)
    outputTable = pd.concat([col1, col2, col3, col4], axis=1, ignore_index=True) 
    outputTable.columns = ['Full Indices', 'Diagnosis', 'Subset Indices', 'Summed Corr']      

    return badRowsTable, outputTable, tempMean, tempStdDev, threshold
    
def breastCancer_processOutliers2(resultImage, sortedData, numFeatures, stdDevThresholdMult):
    # The following finds a subset of rows from the input table that correspond to a given range of covariance values
    # -- resultImage was formed from a correlation image of subject vectors of features -- could be for "B" or "M"
    # -- sortedData is sorted based on "B" than "M" subjects and has a index column into the original data table 
    #    -- 1st column is "indices" and 2nd column is "Diagnosis" and remaining numFeatures columns have the feature data
    # -- find the subjects resulting in high scores 

    # Get the stats
    row_sums = resultImage.sum(axis=1)
    tempMean = statistics.mean(row_sums)
    tempStdDev = statistics.stdev(row_sums)
    threshold1 = tempMean - 0.25*tempStdDev
    threshold2 = tempMean + 0.25*tempStdDev
    threshold3 = tempMean + stdDevThresholdMult*tempStdDev
    
    # Find the "good" rows
    #indices = [i for i, val in enumerate(row_sums) if val < threshold2 and val > threshold1]
    indices = [i for i, val in enumerate(row_sums) if val > threshold3]
    goodRowsTable = sortedData.iloc[indices,:]
    col1 = goodRowsTable.iloc[:,0].reset_index(drop=True)
    col1 = pd.DataFrame(col1)
    col2 = goodRowsTable.iloc[:,1].reset_index(drop=True)
    col2 = pd.DataFrame(col2)
    col3 = pd.DataFrame(indices, columns=['Image Indices']).reset_index(drop=True)
    col4 = row_sums[indices]
    col4 = pd.DataFrame(col4).reset_index(drop=True)
    outputTable = pd.concat([col1, col2, col3, col4], axis=1, ignore_index=True) 
    outputTable.columns = ['Full Indices', 'Diagnosis', 'Subset Indices', 'Summed Corr']      

    return goodRowsTable, outputTable, tempMean, tempStdDev, threshold3
    
def breastCancer_crossCorImage1(X, y, numFeatures, featureSpaceFlag, splitSubjectsFlag, displayImagesFlag, saveImages, outputDirectory):
    # The following creates a unit vector for each column in X
    # -- use this to create a covariance matrix across subjects [569x569]
    # -- ie vector normalize each subject

    # Preprocessing note (5/19/26)
    # -- the table.corr() ignores the input scaling and rescales so that the max value on the diagonal is 1
    # -- the traces seen in sortedData, fullData, and dataFinal will NOT correspond to the traces used in 
    #    the resultImages and these traces will use the vector normalization defined below
    # -- note that this scaling is on the full data set before separating into the benign and malignant subjects
    X =  X.iloc[:,0:numFeatures]
    scaledArray = normalize(X, axis=0)

    # Create the data frame 
    scaledData = pd.DataFrame(scaledArray, columns=X.columns)
    fullData = pd.concat([y,scaledData], axis=1)
    numRows, numCols = fullData.shape
    indicesTable = pd.DataFrame(range(0, numRows), columns=['indices'])
    fullData = pd.concat([indicesTable,fullData], axis=1)
    
    # Sort by the 'Diagnosis' column
    sortedData = fullData.sort_values(by=['Diagnosis','indices'], ascending=[True, True],ignore_index=True)
    indicesB = sortedData.index[sortedData["Diagnosis"] == "B"].tolist()
    indicesM = sortedData.index[sortedData["Diagnosis"] == "M"].tolist()
    dataFinal = sortedData.drop(columns=['indices','Diagnosis'])
    dataFinal = dataFinal.iloc[:,range(0,numFeatures)]
    
    # Two cases -- use all subjects or split the matrix based on 'Diagnosis'
    dataImage1 = []
    dataImage2 = []
    resultImage = []
    resultImage1 = []
    resultImage2 = []
    if (splitSubjectsFlag==1):
        dataImage1 = dataFinal.iloc[indicesB,:]
        if (featureSpaceFlag==0):
            dataImage1 = dataImage1.T
        resultImage1 = dataImage1.corr()
        if (displayImagesFlag==1):
            plt.imshow(resultImage1, cmap='viridis')
            plt.colorbar(label='Benign Intensity Scale') # Adds the vertical scale bar
            if (saveImages==1):
                fullFileName = os.path.join(outputDirectory, 'BenignCrossCorrImage.jpg')
                plt.savefig(fullFileName,dpi=600, bbox_inches='tight')
            plt.show()

        dataImage2 = dataFinal.iloc[indicesM,:]
        if (featureSpaceFlag==0):
            dataImage2 = dataImage2.T
        resultImage2 = dataImage2.corr()
        if (displayImagesFlag==1):
            plt.imshow(resultImage2, cmap='viridis')
            plt.colorbar(label='Malignant Intensity Scale') # Adds the vertical scale bar
            if (saveImages==1):
                fullFileName = os.path.join(outputDirectory, 'MalignantCrossCorrImage.jpg')
                plt.savefig(fullFileName,dpi=600, bbox_inches='tight')            
            plt.show()
        
    else:
        if (featureSpaceFlag==0):
            dataFinal = dataFinal.T
        resultImage = dataFinal.corr()
        if (displayImagesFlag==1):        
            plt.imshow(resultImage, cmap='viridis')
            plt.colorbar(label='Intensity Scale') # Adds the vertical scale bar
            if (saveImages==1):
                fullFileName = os.path.join(outputDirectory, 'CrossCorrImage.jpg')
                plt.savefig(fullFileName,dpi=600, bbox_inches='tight')                
            plt.show()
    
    return fullData, sortedData, dataFinal, dataImage1, dataImage2, resultImage, resultImage1, resultImage2, indicesB, indicesM
    
def breastCancer_featurePlot(X, y, plot_pairs, columnNames):
    # Returns without generating plots if the number of subplots is not 9
    
    # Scale the X data
    # -- Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaledArray = scaler.fit_transform(X)
    scaledData = pd.DataFrame(scaledArray, columns=X.columns)
    fullData = pd.concat([y,scaledData], axis=1)

    # Do a multi-plot
    # -- How many subplots
    errorFlag = 0
    numSubplots = len(plot_pairs)
    if (numSubplots==9):
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 8))
        axes_flat = axes.flatten()
    else:
        errorFlag = 1
        return errorFlag
        
    for index, (index1, index2) in enumerate(plot_pairs):
        axis1_label = columnNames[index1-1]
        axis2_label = columnNames[index2-1]
        sns.scatterplot(data=fullData, x=axis1_label, y=axis2_label, hue="Diagnosis", ax=axes_flat[index])
        
    plt.tight_layout()
    plt.show()

    return errorFlag
    
def pls_ScatterPlots(Scores, y, errorList, saveOutput, outputDirectory):
    # Assemble the data
    fullData = pd.concat([y,Scores], axis=1)
    fullData.loc[errorList, 'Diagnosis'] = 'E'
    
    # Do a multi-plot
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))
    axes_flat = axes.flatten()
    
    plot_pairs = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    columnNames = ["PLS-1", "PLS-2", "PLS-3", "PLS-4", "PLS-5"]
    for index, (index1, index2) in enumerate(plot_pairs):
        axis1_label = columnNames[index1-1]
        axis2_label = columnNames[index2-1]
        sns.scatterplot(data=fullData, x=axis1_label, y=axis2_label, hue="Diagnosis", ax=axes_flat[index])
    
    if (saveOutput):
        fullFileName = os.path.join(outputDirectory, 'PLS_ScatterPlots.jpg')
        plt.savefig(fullFileName,dpi=600, bbox_inches='tight')  
        
    plt.tight_layout()
    plt.show() 

def pls_3d_ScatterPlot(Scores, y, subjectsTable, errorList):
    # Assemble the data
    fullData = pd.concat([y,y,subjectsTable,Scores], axis=1)
    columnNames = ["Diagnosis","Orig Diag","Subjects","PLS-1", "PLS-2", "PLS-3", "PLS-4", "PLS-5"]
    fullData.columns = columnNames
    fullData.loc[errorList, 'Diagnosis'] = 'E'

    # Generate the plot
    # -- columnNames = ["PLS-1", "PLS-2", "PLS-3", "PLS-4", "PLS-5"]
    fig = px.scatter_3d(
        fullData,
        x='PLS-1',
        y='PLS-2',
        z='PLS-3',
        color='Diagnosis',     
        title="Full Feature + Engineered Features Model",
        width=1000,   # Set explicit pixel width
        height=800,   # Set explicit pixel height    
        size_max=1,   # Keeps all markers strictly at a smaller scale
        hover_data=['Subjects', 'Orig Diag']
    )

    # Render the plot
    fig.show()

def calculate_pls_SSY(binary_target, n_components, x_scores, y_loadings):
    """
    Calculate SSQ
    """

    # Total variance
    y_centered = binary_target - np.mean(binary_target)
    ssy_total = np.sum(y_centered**2)

    # Explained Sum of Squares for each Component    
    ssy_explained = []
    for i in range(n_components):
        # Each component's fit is the outer product of scores and loadings
        y_pred = np.outer(x_scores[:,i], y_loadings[:,i])
        ss_i = np.sum(y_pred**2)
        ssy_explained.append(ss_i)

    # Total explained
    ss_total_explained = sum(ssy_explained)
    ss_residual = ssy_total - ss_total_explained
    var_explained_ratio = [ss / ssy_total for ss in ssy_explained]

    return ssy_total, ss_total_explained, ssy_explained, var_explained_ratio

def calculate_pls_SSX(model, X, n_components, x_scores, x_loadings):
    """
    Calculate SSQ
    """

    # Total variance
    scaler = model.named_steps['scaler']
    X_scaled = scaler.transform(X)
    XX_Cov = X_scaled.T @ X_scaled
    s = np.diag(XX_Cov)
    ssx_total = np.sum(s)

    # Explained Sum of Squares for each Component
    # Formula: p.T * p (X loadings squared norm) multiplied by the score variance
    ssx_explained = []
    for i in range(n_components):
        # Reconstructing the component's shared variance
        score_var = np.sum(x_scores[:, i]**2)
        loading_norm = np.sum(x_loadings[:, i]**2)
        
        # In PLS, score variance * loading squared norm yields the SS explained
        comp_ss = score_var * loading_norm
        ssx_explained.append(comp_ss)
    
    # Calculate Percentage of Variance Explained
    var_explained_ratio = [ss / ssx_total for ss in ssx_explained]

    return ssx_total, ssx_explained, var_explained_ratio
    
def calculate_vip(x_scores, x_weights, y_loadings):    
    """
    Calculate VIP scores for a fitted scikit-learn PLSRegression model.
    """
    t = x_scores
    w = x_weights
    q = y_loadings
    
    #m, p = model.x_weights_.shape[0], model.x_weights_.shape[0]
    m, p = x_weights.shape[0], x_weights.shape[0]
    _, h = t.shape
    vips = np.zeros((p,))
    
    # Calculate the explained variance for each component
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        # Weighting for each variable across components
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        
    return vips

def perform_crossValidation(model, X, binary_target, saveOutput, outputDirectory, titlePrefix):
    #
    # Calculate loo cross-validation and use the scores in discriminant analysis for each model
    # -- binary_target contains 0 for class0 (benign for example) and 1 for class1 (malignant for example)
    #
    
    # Get a set of predicted scores using loo cross-validation
    scores = []
    probs1 = []
    probs2 = []
    preds = []
    gt = []
    numSamples = X.shape[0]
    numFeatures = X.shape[1]
    for i in range(numSamples):
        # Pick the data for the currrent loo and build the model
        data1 = X[:i]
        data2 = X[i+1:]
        newData = pd.concat([data1,data2],ignore_index=True)
        newTarget = binary_target[:]
        del newTarget[i]
        currentRow = X.iloc[[i]]
        currentTarget = binary_target[i]
        model.fit(newData,newTarget)
        
        # Find the scores on the training data and the resulting discriminator
        # -- calculates the probability (density) for the two class distributions
        # -- the target point belongs to the distribution resulting in the largest probability density value
        modelScores = model.predict(newData)
        list1 = [i for i, val in enumerate(newTarget) if val==0]
        list2 = [i for i, val in enumerate(newTarget) if val==1]
        mu0, std0 = norm.fit(modelScores[list1])
        mu1, std1 = norm.fit(modelScores[list2])        
    
        # Predict the score for the loo target
        currentScore = model.predict(currentRow)
        prob0 = norm.pdf(currentScore, mu0, std0)
        prob1 = norm.pdf(currentScore, mu1, std1)
        currentPred = (prob1 > prob0).astype(int)

        # Update the arrays for return                         
        scores.append(currentScore)
        probs1.append(prob0)
        probs2.append(prob1)
        preds.append(currentPred)
        gt.append(currentTarget)

    # Repackage scores
    scoresList = np.concatenate(scores).tolist()
    probsList1 = np.concatenate(probs1).tolist()
    probsList2 = np.concatenate(probs2).tolist()
    predsList = np.concatenate(preds).tolist()

    # Calculate some stats
    trueCt, tpCt, tnCt, fpCt, fnCt = calculatePredictionStats(preds, gt)
    cvError = (fpCt + fnCt)/numSamples

    # Output some stats
    if (saveOutput==1):
        data = {'Results': [numSamples, numFeatures, tpCt, tnCt, fpCt, fnCt, cvError]}
        tempTitle = titlePrefix + '_PLSR_Stats.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        df = pd.DataFrame(data, index=['Num Subjects', 'Num Features', 'TP Ct', 'TN Ct', 'FP Ct', 'FN Ct', 'Model Error'])
        df.to_excel(fullFileName)    
    
    return scoresList, probsList1, probsList2, predsList, gt, trueCt, tpCt, tnCt, fpCt, fnCt, cvError

def calculatePredictionStats(predsList, gt):
    # Calculate some stats
    
    df = pd.DataFrame(list(zip(predsList, gt)), columns=['Preds', 'GT'])
    tpCt = ((df['Preds'] == 1) & (df['GT'] == 1)).sum()
    tnCt = ((df['Preds'] == 0) & (df['GT'] == 0)).sum()
    trueCt = tpCt + tnCt
    fnCt = ((df['Preds'] == 0) & (df['GT'] == 1)).sum()
    fpCt = ((df['Preds'] == 1) & (df['GT'] == 0)).sum()    

    return trueCt, tpCt, tnCt, fpCt, fnCt
    
def discriminantPlot(dataTable, saveOutput, outputDirectory, titlePrefix):
    #
    # Show a scatter plot of points ordered by class (gt)
    # -- highlight misclassifications in red
    # -- dataTable = pd.DataFrame({'Subjects': indices, 'GT': gt,'Scores': scores, 'Probs1': probs1, 'Probs2': probs2, 'Preds': preds})

    # Sort and organize the data
    # -- add some columns
    df = dataTable.copy()
    numObjects = len(df)
    df['Error1'] = 0
    df['Error2'] = 0
    df['Error'] = 0
    filtered1_indices = df[(df['GT'] != df['Preds']) & (df['GT'] == 1)].index.tolist()
    filtered2_indices = df[(df['GT'] != df['Preds']) & (df['GT'] == 0)].index.tolist()
    filtered3_indices_all = filtered1_indices + filtered2_indices
    filtered3_indices = list(set(filtered3_indices_all))
    
    df.loc[filtered1_indices,'Error1'] = 1
    df.loc[filtered2_indices,'Error2'] = 1
    df.loc[filtered3_indices,'Error'] = 1
    sorted_df = df.sort_values(by='GT', ignore_index=True)
    sorted_df['SortOrder'] = range(1,numObjects+1)

    # Get errors
    error_rows = sorted_df[sorted_df['Error'] == 1]

    # save plot, etc
    if (saveOutput==1):
        # Create and save plot
        sns.scatterplot(data=sorted_df,x='SortOrder',y='Scores',hue='Error')
        tempTitle = titlePrefix + '_DiscriminationPlot.jpg' 
        fullFileName = os.path.join(outputDirectory, tempTitle)
        plt.savefig(fullFileName,dpi=600, bbox_inches='tight')
        
        # Save full table
        tempTitle = titlePrefix + '_PLSR_DiscriminantOutputSorted.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        sorted_df.to_excel(fullFileName)

        # Save error rows
        tempTitle = titlePrefix + '_DiscriminationErrors.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        error_rows.to_excel(fullFileName)

    return sorted_df, error_rows
    
def predictProbabiltyFromScores(scores, binary_target, testScores):
    #

    # Find the indices for each of the two classes
    # -- then calculate the distributions for the scores for those two classes
    list1 = [i for i, val in enumerate(binary_target) if val ==0]
    list2 = [i for i, val in enumerate(binary_target) if val==1]
    mu0, std0 = norm.fit(scores[list1])
    mu1, std1 = norm.fit(scores[list2])

    # Convert the testScores to probabilities
    probs0 = norm.pdf(testScores, mu0, std0)
    probs1 = norm.pdf(testScores, mu1, std1)
    testPreds = (probs1 > probs0).astype(int)  

    return probs0, probs1, testPreds
    
def vary_PlsFactors(X, binary_target, maxFactors, saveOutput, outputDirectory, titlePrefix):
    #
    calibrationError = []
    crossvalError = []
    numSamples = X.shape[0]
    for i in range(2, maxFactors+1):
        # Define the model
        model = Pipeline([
        ('scaler', StandardScaler()),
         ('pls', PLSRegression(n_components=i))
        ])

        # Build the loo cross-val model
        saveOutput = 0
        scores, probsList1, probsList2, predsList, gt, trueCt, tpCt, tnCt, fpCt, fnCt, cvError = perform_crossValidation(model, X, binary_target, \
                saveOutput, outputDirectory, titlePrefix)
        crossvalError.append(cvError)

        # Get the calibration error
        # -- build the full model and predict on it
        model.fit(X,binary_target)
        predScores = model.predict(X)
        probs0, probs1, predProbs = predictProbabiltyFromScores(predScores, binary_target, predScores)
        trueCt, tpCt, tnCt, fpCt, fnCt = calculatePredictionStats(predProbs, binary_target)
        calError = (fpCt + fnCt)/numSamples
        calibrationError.append(calError)
        
    return crossvalError, calibrationError

def apply_PLSDA_Model(model, X, binary_target, subjectLabelTable, saveOutput, outputDirectory, titlePrefix):
    # Apply a PLSDA model that has already been built
    
    # Predict using the full model
    numSamples = X.shape[0]
    numFeatures = X.shape[1]
    predScores = model.predict(X)
    probs0, probs1, predProbs = predictProbabiltyFromScores(predScores, binary_target, predScores)    
    trueCt, tpCt, tnCt, fpCt, fnCt = calculatePredictionStats(predProbs, binary_target)
    modelError = (fpCt + fnCt)/numSamples    

    # Output a scores file 
    # -- also used as input to discriminant plot
    # -- note that 'Subjects' is offset by 1 and cannot therefore serve directly as an index into a table that is 0s-based
    indices = list(range(1,numSamples+1))
    dataTable = pd.DataFrame({'Indices': subjectLabelTable['Indices'], 'Subjects': subjectLabelTable['Subjects'], 'GT': binary_target, \
                              'Scores': predScores, 'Probs1': probs0, 'Probs2': probs1, 'Preds': predProbs})
    tempTitle = titlePrefix + '_AppliedModel_ScoresOutput.xlsx'
    fullFileName = os.path.join(outputDirectory, tempTitle)
    dataTable.to_excel(fullFileName)

    # Generate the discriminant plot
    sorted_df, errors = discriminantPlot(dataTable, saveOutput, outputDirectory, titlePrefix)

    # Save output
    if (saveOutput==1):       
        # Output some stats
        data = {
            'Results': [numSamples, numFeatures, tpCt, fpCt, fnCt, modelError]
        }
        tempTitle = titlePrefix + '_PLSR_Stats.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        df = pd.DataFrame(data, index=['Num Subjects', 'Num Features', 'TP Ct', 'FP Ct', 'FN Ct', 'Model Error'])
        df.to_excel(fullFileName)
        
    return modelError, sorted_df, errors
    
def build_PLSDA_Model(X, binary_target, subjectLabelTable, numFactors, saveOutput, outputDirectory, titlePrefix):
    #

    # Build the full model
    model = Pipeline([
        ('scaler', StandardScaler()),
         ('pls', PLSRegression(n_components=numFactors))
    ])
    
    # Build the full model
    model.fit(X,binary_target)
    n_components = model.named_steps['pls'].n_components
    x_loadings = model.named_steps['pls'].x_loadings_
    x_scores = model.named_steps['pls'].x_scores_
    y_scores = model.named_steps['pls'].y_scores_
    x_weights = model.named_steps['pls'].x_weights_
    y_loadings = model.named_steps['pls'].y_loadings_
    model_coefs = model.named_steps['pls'].coef_
    model_intercept = model.named_steps['pls'].intercept_
    
    # Calculate some metrics from the full model
    eigenvalues = np.var(x_scores,axis=0)
    ssx_total, ssx_explained, ssx_var_explained_ratio = calculate_pls_SSX(model, X, n_components, x_scores, x_loadings)
    ssy_total, ssy_total_explained, ssy_explained, ssy_var_explained_ratio = calculate_pls_SSY(binary_target, n_components, x_scores, y_loadings)
    ssx_summed = np.sum(ssx_var_explained_ratio)
    ssy_summed = np.sum(ssy_var_explained_ratio)
    ssx_percent = (ssx_var_explained_ratio/ssx_summed)*100.0
    ssy_percent = (ssy_var_explained_ratio/ssy_summed)*100.0
    
    # Calculate the VIP scores from the full model
    vip_scores = calculate_vip(x_scores, x_weights, y_loadings)

    # Predict using the full model
    numSamples = X.shape[0]
    numFeatures = X.shape[1]
    predScores = model.predict(X)
    probs0, probs1, predProbs = predictProbabiltyFromScores(predScores, binary_target, predScores)    
    trueCt, tpCt, tnCt, fpCt, fnCt = calculatePredictionStats(predProbs, binary_target)
    modelError = (fpCt + fnCt)/numSamples    

    # Output a scores file 
    # -- also used as input to discriminant plot
    # -- note that 'Subjects' is offset by 1 and cannot therefore serve directly as an index into a table that is 0s-based
    indices = list(range(1,numSamples+1))
    dataTable = pd.DataFrame({'Indices': subjectLabelTable['Indices'], 'Subjects': subjectLabelTable['Subjects'], 'GT': binary_target, \
                              'Scores': predScores, 'Probs1': probs0, 'Probs2': probs1, 'Preds': predProbs})
    if (saveOutput==1):
        tempTitle = titlePrefix + '_FullModel_ScoresOutput.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        dataTable.to_excel(fullFileName)

    # Generate the discriminant plot
    sorted_df, errors = discriminantPlot(dataTable, saveOutput, outputDirectory, titlePrefix)

    # Save output
    if (saveOutput==1):       
        # Output a PLSR metrics file
        df = pd.DataFrame({'SSX': ssx_percent,'SSY': ssy_percent, 'Eigenvalues': eigenvalues})
        tempTitle = titlePrefix + '_PLSR_MetricsOutput.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        df.to_excel(fullFileName)
        
        # Output a PLSR VIP scores file
        colNames = X.columns.tolist()
        df = pd.DataFrame({'Features': colNames, 'VIP': vip_scores})
        tempTitle = titlePrefix + '_PLSR_VipScores.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        df.to_excel(fullFileName) 

        # Output some stats
        data = {
            'Results': [numSamples, numFeatures, tpCt, fpCt, fnCt, modelError]
        }
        tempTitle = titlePrefix + '_PLSR_Stats.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        df = pd.DataFrame(data, index=['Num Subjects', 'Num Features', 'TP Ct', 'FP Ct', 'FN Ct', 'Model Error'])
        df.to_excel(fullFileName)
        
    return model, modelError, sorted_df, errors
    
def outlier_Plots_Script(inputData, errors, sorted_df, normFlag, saveImages, showOutlierTracesFlag, plotLegend, outputDirectory, tracePlotOffset, titlePrefix):
    # Inputs
    #   inputData -- the same data used to build models prior to this call
    #   errors -- the errors table returned by crossVal_BuildModel_Script or build_PLSDA_Model, which is expected to have a specific format
    #   sorted_df -- the table of filtered data returned by the two above functions, both of which generate this table in the discriminantPlot function
    #             -- SortOrder contains the ordering in sorted_df
    # Edits
    #   errors -- now includes a Subjects column and in Indices column that points to the order in the discriminant plot
    #   updated scaling and indices extraction 

    
    # Normalize the data over columns 
    scaledArray = normalize(inputData, axis=0)
    scaledData = pd.DataFrame(scaledArray, columns=inputData.columns)
    
    # Scale the data over columns
    scaler = StandardScaler()
    scaledArray = scaler.fit_transform(scaledData)
    scaledData = pd.DataFrame(scaledArray, columns=inputData.columns)
    
    # Get outliers
    errors_FPs = errors.loc[errors['Error2'] == 1]
    errors_FNs = errors.loc[errors['Error1'] == 1]

    # 1st plot -- offset spectra
    plotLabel = 'FP Feature Traces'
    fileName = titlePrefix + '_FPsFeatureTracesOffset.jpg'
    errors_FPs = errors_FPs.sort_values(by='Subjects')
    fpSubjects = errors_FPs['Subjects'].tolist()
    fpIndices = errors_FPs['Indices']
    fpRowsDataTable = scaledData.iloc[fpIndices,:]   
    if len(fpRowsDataTable)!=0:
        breastCancer_createPlot4(fpRowsDataTable, fpSubjects, plotLabel, saveImages, normFlag, outputDirectory, plotLegend, fileName, tracePlotOffset)
    
    # 2nd plot -- offset spectra
    plotLabel = 'FN Feature Traces'
    fileName = titlePrefix + '_FNsFeatureTracesOffset.jpg'
    errors_FNs = errors_FNs.sort_values(by='Subjects')
    fnSubjects = errors_FNs['Subjects'].tolist()
    fnIndices = errors_FNs['Indices']
    fnRowsDataTable = scaledData.iloc[fnIndices,:]
    if len(fnRowsDataTable)!=0:
        breastCancer_createPlot4(fnRowsDataTable, fnSubjects, plotLabel, saveImages, normFlag, outputDirectory, plotLegend, fileName, tracePlotOffset)

def crossVal_VaryFactors_Script(X, binary_target, maxFactors, outputDirectory, plotOutput, saveOutput, titlePrefix):
    #
    
    # Vary the number of factors from 2 to maxFactors to build PSLDA models
    crossvalError, calibrationError = vary_PlsFactors(X, binary_target, maxFactors, saveOutput, outputDirectory, titlePrefix)
    
    # Create a plot
    if (plotOutput==1):
        fig, ax = plt.subplots()
        xAxisValues = [x for x in range(2,maxFactors+1)]
        ax.plot(xAxisValues, calibrationError, marker='o', linestyle='-', color='b', label='Calibration Error')
        ax.plot(xAxisValues, crossvalError, marker='s', color='green', label='Cross-Validation Error')
        ax.set_ylim(0.0, 0.1)
        ax.set_xlabel('Num PLS Factors')
        ax.set_ylabel('Model Error')
        ax.set_title('Factor Selection')
        plt.legend(fontsize=8)
        if (saveOutput==1):
            tempTitle = titlePrefix + '_Error_vs_Factors.jpg'
            fullFileName = os.path.join(outputDirectory, tempTitle)
            plt.savefig(fullFileName,dpi=600, bbox_inches='tight')  
        plt.show() 
    
    # Generate a table for output
    df = pd.DataFrame({'Factors': xAxisValues,'Cal Error': calibrationError, 'CrossVal Error': crossvalError})
    if (saveOutput==1):
        tempTitle = titlePrefix + '_Error_vs_Factors.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        df.to_excel(fullFileName, index=False)
        
    return df, crossvalError, calibrationError

def crossVal_BuildModel_Script(X, y, subjectLabelTable, binary_target, numFactors, saveOutput, plotNormFlag, outputDirectory, titlePrefix):
    #
    
    # Build a model using the number of factors determined to be appropriate
    model = Pipeline([
    ('scaler', StandardScaler()),
     ('pls', PLSRegression(n_components=numFactors))
    ])
    
    # Build the loo cross-val model
    scores, probs1, probs2, preds, gt, trueCt, tpCt, tnCt, fpCt, fnCt, cvError = perform_crossValidation(model, X, binary_target, saveOutput,\
                                                                                    outputDirectory, titlePrefix)
    
    # Output a scores file 
    # -- note that 'Subjects' is offset by 1 
    # -- using a new table with 'Indices' and 'Subjects' columns
    numSamples = len(binary_target)
    dataTable = pd.DataFrame({'Indices': subjectLabelTable['Indices'], 'Subjects': subjectLabelTable['Subjects'], 'GT': gt,\
                              'Scores': scores, 'Probs1': probs1, 'Probs2': probs2, 'Preds': preds})
    if (saveOutput==1):
        tempTitle = titlePrefix + '_PLSR_ScoresOutput.xlsx'
        fullFileName = os.path.join(outputDirectory, tempTitle)
        dataTable.to_excel(fullFileName)
    
    # Generate a discriminant plot and some stats
    sorted_df, errors = discriminantPlot(dataTable, saveOutput, outputDirectory, titlePrefix)

    # Create a plot of the outliers and an avg of the inliers
    # -- initialize scaledData since it must be returned even if saveOutput==0
    scaledData = []
    if (saveOutput==1):
        scaledData, *_ = breastCancer_createPlot2(X, y, sorted_df, saveOutput, plotNormFlag, outputDirectory, titlePrefix)
        
    return scaledData, sorted_df, errors, trueCt, tpCt, tnCt, fpCt, fnCt, cvError

def generateCombinationsList(removalList, lineLimit, depthLimit):
    all_combinations = []
    for i in range(len(removalList) + 1):
        if (i<=depthLimit):
            combinations_object = itertools.combinations(removalList, i)
            combinations_list = list(combinations_object)
            all_combinations.extend(combinations_list)
    
    all_combinations_trunc = all_combinations[0:lineLimit]    
    
    return all_combinations_trunc

def createSubjectLabelsTable(subjectsTable, removalList):
    # This function creates a table that:
    # -- keeps the original subject #s
    # -- creates a parallel column that records the new indices 
    #    into the table after the removal of the requested subjects
      
    # Remove requested subjects
    newSubjectsTable = subjectsTable.copy()    
    newSubjectsTable = newSubjectsTable.drop(removalList).reset_index(drop=True)

    # Generate the new indices and create the final table 
    numSubjects = len(newSubjectsTable)
    indicesList = [x for x in range(0,numSubjects)]
    indicesTable = pd.DataFrame(indicesList, columns=['Indices'])
    newSubjectLabels = pd.concat([indicesTable, newSubjectsTable], axis=1)  

    return newSubjectLabels

def subjectRemovalScript(removalList, all_combinations, lineLimit, depthLimit, inputData, y, subjectsTable, numFactors, saveOutput, \
                         plotNormFlag, outputDirectory, titlePrefix):
    # This function loops over all entries in removalList, creates all possible combinations features to remove,
    # builds each model and generates some stats to return
    # -- it is expected that there are less entries in removal list than rows in inputData
    
    # Generate the list of all combinations
    # -- only generate this if it wasn't passed in 
    if not all_combinations:
        all_combinations = generateCombinationsList(removalList, lineLimit, depthLimit)
    
    # Create a subject table to keep track of removed subjects per model
    # -- column headers have the subject #s (ie, increment the subject table indices by 1)
    maxRemovals = len(removalList)
    numExpts = len(all_combinations)
    removedSubjectsArray = np.zeros((numExpts, maxRemovals+1))
    removedListSubjects = [x + 1 for x in removalList] 
    columnNames = [str(num) for num in removedListSubjects]
    columnNames.append("Cal Error")
    removedSubjectsTable = pd.DataFrame(removedSubjectsArray,columns=columnNames)

    # Initialize the subject table required by certain functions
    numSamples = len(y)
    inputLabels = list(range(1,numSamples+1))
    subjectsTable = pd.DataFrame({'Subjects': inputLabels})    

    # Initialize the display handle for progress updates to Jupyter
    update_handle = display("Current iteration:   ", display_id=True)
    
    # Loop over all combinations
    cvErrorList = []
    modelErrorList = []
    combinationsListStr = []
    trueCtList = []
    fpCtList = []
    fnCtList = []
    for i in range(numExpts):
        # Update Jupyter Notebook
        if ((i % 5)==0):
            update_handle.update(f"Current iteration:   {i} of {numExpts}")
        
        curList = list(all_combinations[i])
        curListSubjects = [x + 1 for x in curList]        
        curListStr = ", ".join(str(num) for num in curListSubjects)
        combinationsListStr.append(curListStr)
        if not curList:
            # empty list -- don't remove any descriptors
            newData = inputData.copy() 
            newY = y.copy()
            newSubjects = subjectsTable.copy()
            target = np.ravel(newY)
            binary_target = [0 if item == "B" else 1 for item in target]
            newSubjectLabels = createSubjectLabelsTable(subjectsTable, [])                 
        else:
            # there are >= 1 descriptors to remove
            newData = inputData.copy() 
            newData = newData.drop(curList)
            newY = y.copy()
            newY = newY.drop(curList)
            target = np.ravel(newY)
            binary_target = [0 if item == "B" else 1 for item in target]
            # Update the indice/subject table
            newSubjectLabels = createSubjectLabelsTable(subjectsTable, curList) 

            
        # Build the model with cross-validation
        # -- note that "scaledData" is empty if tempSaveOutput==0, which is fine since it's not needed
        tempSaveOutput = 0
        tempPlotNormFlag = 0
        scaledData, sorted_df, errors, trueCt, tpCt, tnCt, fpCt, fnCt, cvError = crossVal_BuildModel_Script(newData, newY, newSubjectLabels, \
                    binary_target, numFactors, tempSaveOutput, tempPlotNormFlag, outputDirectory, titlePrefix)

        # Build the model without cross-validation
        tempSaveOutput = 0
        modelStructure, modelError, sorted_df, errors = build_PLSDA_Model(newData, binary_target, newSubjectLabels, numFactors, \
                    tempSaveOutput, outputDirectory, titlePrefix)

        # Increment the counts for the current subjects in removedSubjectsTable
        curListSubjectsStrings = [str(x) for x in curListSubjects]
        removedSubjectsTable.loc[i, curListSubjectsStrings] += 1
        removedSubjectsTable.iloc[i,maxRemovals] = modelError
        
        # gather some stats to return for each expt
        cvErrorList.append(cvError)
        modelErrorList.append(modelError)
        trueCtList.append(trueCt)
        fpCtList.append(fpCt)
        fnCtList.append(fnCt)

        # Save some intermediate copies
        if ((i % 200)==0):
            # Generate a full data table to return
            combinationTable = pd.DataFrame(combinationsListStr)
            trueCtTable = pd.DataFrame(trueCtList)
            fpCtTable = pd.DataFrame(fpCtList)
            fnCtTable = pd.DataFrame(fnCtList)
            cvErrorTable = pd.DataFrame(cvErrorList)
            calErrorTable = pd.DataFrame(modelErrorList)    
            returnTable = pd.concat([combinationTable, trueCtTable, fpCtTable, fnCtTable, cvErrorTable, calErrorTable], axis=1)
            colNames = ['Subjects Removed','True Ct', 'FP ct', 'FN Ct','CV Error','Cal Error']
            returnTable.columns = colNames    
            
            # Always save the following tables
            tempTitle = titlePrefix + '_RemoveSubjectsSummaryTable.xlsx'
            fullFileName = os.path.join(outputDirectory, tempTitle)
            returnTable.to_excel(fullFileName, index=False)
            tempTitle = titlePrefix + '_RemovedSubjectsTable.xlsx'
            fullFileName = os.path.join(outputDirectory, tempTitle)
            removedSubjectsTable.to_excel(fullFileName, index=False)        

    # Generate a full data table to return
    combinationTable = pd.DataFrame(combinationsListStr)
    trueCtTable = pd.DataFrame(trueCtList)
    fpCtTable = pd.DataFrame(fpCtList)
    fnCtTable = pd.DataFrame(fnCtList)
    cvErrorTable = pd.DataFrame(cvErrorList)
    calErrorTable = pd.DataFrame(modelErrorList)    
    returnTable = pd.concat([combinationTable, trueCtTable, fpCtTable, fnCtTable, cvErrorTable, calErrorTable], axis=1)
    colNames = ['Subjects Removed','True Ct', 'FP ct', 'FN Ct','CV Error','Cal Error']
    returnTable.columns = colNames    
    
    # Always save the following tables
    tempTitle = titlePrefix + '_RemoveSubjectsSummaryTable.xlsx'
    fullFileName = os.path.join(outputDirectory, tempTitle)
    returnTable.to_excel(fullFileName, index=False)
    tempTitle = titlePrefix + '_RemovedSubjectsTable.xlsx'
    fullFileName = os.path.join(outputDirectory, tempTitle)
    removedSubjectsTable.to_excel(fullFileName, index=False)
        
    return all_combinations, cvErrorList, modelErrorList, combinationsListStr, returnTable, removedSubjectsTable
        
def crossCorrImage_FeatureScript(X, y, numFeatures, splitSubjectsFlag, displayImagesFlag, saveImages, outputDirectory):

    # Call the main function
    # -- featureSpaceFlag = 1 generates an image that is of size nxn features
    featureSpaceFlag = 1
    fullData, sortedData, dataFinal, dataImage1, dataImage2, resultImage, resultImage1, resultImage2, indicesB, indicesM = \
        breastCancer_crossCorImage1(X, y, numFeatures, featureSpaceFlag, splitSubjectsFlag, displayImagesFlag, saveImages, outputDirectory)
   
    return fullData, sortedData, dataFinal, dataImage1, dataImage2, resultImage, resultImage1, resultImage2, indicesB, indicesM
        
def crossCorrImage_SubjectScript(X, y, numFeatures, splitSubjectsFlag, displayImagesFlag, saveImages, outputClassFlag, 
    stdDevThresholdMult1, stdDevThresholdMult2, displayPlotsFlag, outputDirectory):

    # Call the main function
    # -- featureSpaceFlag = 0 generates an image that is of size mxm subjects
    # -- the function prepends an 'indices' column and then a 'Diagnosis' column and then sorts by both keys, where "B" is first and then "M"
    # -- fullData is the full data table before sorting
    # -- sortedData is the full data table after sorting (ie with the two prepended columns)
    # -- dataFinal is the sorted data table that also has the prepended columns removed
    # -- dataImage1, resultImage1 are for "B" subjects; dataImage2, resultImage2 are for "M" subjects
    # -- data images are the input data;  result images are the cross-correlation results
    # -- note that the indices allow one to access subjects from 0 to 569 (tot num subjects)
    featureSpaceFlag = 0
    fullData, sortedData, dataFinal, dataImage1, dataImage2, resultImage, resultImage1, resultImage2, indicesB, indicesM = \
        breastCancer_crossCorImage1(X, y, numFeatures, featureSpaceFlag, splitSubjectsFlag, displayImagesFlag, saveImages, outputDirectory)

    # No splitting results in empty images for dataImage1, dataImage2, resultImage1 and resultImage2
    # -- return with only the results applicable to the combined data
    if (splitSubjectsFlag==0):
        return fullData, sortedData, dataFinal, resultImage, indicesB, indicesM
    
    # sortedData is in the order benign/malignant and must be split into the two matrices for the next functions
    benignCt = len(indicesB)
    malignantCt = len(indicesM)
    sortedDataBenign = sortedData.iloc[0:benignCt,:].reset_index(drop=True)
    sortedDataMalignant = sortedData.iloc[benignCt:,:].reset_index(drop=True)
    # 
    benignLabels = list(range(benignCt))
    resultImage1.columns = benignLabels
    resultImage1.index = benignLabels
    #
    malignantLabels = list(range(malignantCt))
    resultImage2.columns = malignantLabels
    resultImage2.index = malignantLabels
    
    # Find the outliers for selected subjects and plot them
    # -- stdDevThresholdMult1 generates a threshold to select outliers below the threshold
    # -- stdDevThresholdMult2 generates a threshold to select feature traces above the threshold to average
    # -- outputFlag is 0 for Benign and 1 for Malignant
    # -- outputBadTable and outputGoodTable have four columns: a) fullIndices - an index (0 to n-1) into fullData
    #    b) Diagnosis  c) subsetIndices - an index into the correlation images (the subtables)  d) Summed Corr
    # -- badRowsTable and goodRowsTable come from sortedData (ie have two prepended columns -- full indices, diagnosis)
    if (outputClassFlag==0):
        # Find the outliers
        badRowsTable, outputBadTable, tempMean, tempStdDev, thresholdBad = \
            breastCancer_processOutliers1(resultImage1, sortedDataBenign, numFeatures, stdDevThresholdMult1)
        # Find the subjects defining in-class values
        goodRowsTable, outputGoodTable, tempMean, tempStdDev, thresholdGood = \
            breastCancer_processOutliers2(resultImage1, sortedDataBenign, numFeatures, stdDevThresholdMult2)
        # Define labels
        plotLabel = 'Benign Subjects'
        # Save stats table
        if (saveImages==1):
            fullFileName = os.path.join(outputDirectory, 'BenignOutliers.xlsx')
            outputBadTable.to_excel(fullFileName, index=False)
    else:
        # Find the outliers
        badRowsTable, outputBadTable, tempMean, tempStdDev, thresholdBad = \
            breastCancer_processOutliers1(resultImage2, sortedDataMalignant, numFeatures, stdDevThresholdMult1)
        # Find the subjects defining in-class values
        goodRowsTable, outputGoodTable, tempMean, tempStdDev, thresholdGood = \
            breastCancer_processOutliers2(resultImage2, sortedDataMalignant, numFeatures, stdDevThresholdMult2)
        # Define labels
        plotLabel = 'Malignant Subjects'    
        # Save stats table
        if (saveImages==1):
            fullFileName = os.path.join(outputDirectory, 'MalignantOutliers.xlsx')
            outputBadTable.to_excel(fullFileName, index=False)
        
    # Create a plot of the worst line traces and the avg class trace
    if (displayPlotsFlag==1):
        avgClass2Label = 'Avg In-Class Trace'
        fileNameTitle = 'AvgInClassTrace.jpg'
        showOutlierTracesFlag = 1
        plotLegend = 1
        normFlag = 0  # ie, don't normalize the traces for this call
        breastCancer_createPlot1(badRowsTable, goodRowsTable, plotLabel, avgClass2Label, showOutlierTracesFlag, saveImages, \
                                 normFlag, outputDirectory, plotLegend, fileNameTitle)
    
    # Create a histogram of the RowSum results
    if (displayPlotsFlag==1):
        if (outputClassFlag==0):
            resultImage = resultImage1
        else:
            resultImage = resultImage2
        row_sums = resultImage.sum(axis=1)
        plt.hist(row_sums, bins=30, color='skyblue', edgecolor='black')
        if (saveImages==1):
            fullFileName = os.path.join(outputDirectory, 'HistogramPlotImage.jpg')
            plt.savefig(fullFileName,dpi=600, bbox_inches='tight')            
        plt.show()

    return fullData, sortedData, dataFinal, dataImage1, dataImage2, resultImage, resultImage1, resultImage2, indicesB, indicesM, \
        badRowsTable, goodRowsTable, outputBadTable