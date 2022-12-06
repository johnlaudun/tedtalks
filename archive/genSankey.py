# https://gist.github.com/samighi/aed166fd8e6d23aced86ac6930cd404e

def genSankey(df,cat_cols=[],value_cols='',title='Sankey Diagram'):
    # maximum of 6 value cols -> 6 colors
    colorPalette = ['#4B8BBE','#306998','#FFE873','#FFD43B','#646464']
    colorNumList = []
    specials = [". ","> ",">> ",".. ","- "]
    labelList = []
    lableDict = {}
    for i,catCol in enumerate(cat_cols):
        n = len(labelList)
        labelListTemp =  list(enumerate(set(df[catCol].values),start=n))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp
        lableDict[catCol] = dict(labelListTemp)

    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))
    # revese the dict 
    rDict = {}
    for k,v in lableDict.items():
        rDict[k] = {str(v) : k for k,v in v.items()}
    
    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]]*colorNum
        
    # transform df into a source-target pair
    sourceTargetDf = pd.DataFrame()
    for i in range(len(cat_cols)-1):
  
            tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]].copy()
            tempDf[cat_cols[i]] =  tempDf[cat_cols[i]].astype(str)
            tempDf[cat_cols[i+1]] =  tempDf[cat_cols[i+1]].astype(str)
            tempDf['s'],tempDf['d'] = cat_cols[i],cat_cols[i+1]
            tempDf.columns = ['source','target','count','s','t']
            tempDf['label'] = df['label']
            sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
            sourceTargetDf = sourceTargetDf.groupby(['source','target','s','t','label']).agg({'count':'sum'}).reset_index()
        
    # add index for source-target pair
#     sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
#     sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
    display(tempDf)
    display(rDict)
    sourceTargetDf['sourceID'] = sourceTargetDf.apply(lambda x: rDict[x['s']][x['source']],axis=1)
    sourceTargetDf['targetID'] = sourceTargetDf.apply(lambda x: rDict[x['t']][x['target']],axis=1)

    
    # creating the sankey diagram
    data = dict(
        type='sankey',
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = [x[1] for x in labelList],

          color = colorList
        ),
        link = dict(
          source = sourceTargetDf['sourceID'],
          target = sourceTargetDf['targetID'],
          value = sourceTargetDf['count'],
          label = sourceTargetDf['label'],
          color = colorList
        )
      )
    
    layout =  dict(
        title = title,
        font = dict(
          size = 10,
            height = 1250,
        width = 1000
        )
    )
       
    fig = dict(data=[data], layout=layout)
    return fig