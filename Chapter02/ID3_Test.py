def main():
    #Lets Create the test dataset to build our tree
    dataset = {'Name':['Person 1','Person 2','Person 3','Person 4','Person 5','Person 6','Person 7','Person 8','Person 9','Person 10'],
           'Salary':['Low','Med','Med','Med','Med','High','Low','High','Med','Low'],
           'Sex':['Male','Male','Male','Female','Male','Female','Female','Male','Female','Male'],
           'Marital':['Unmarried','Unmarried','Married','Married','Married','Unmarried','Unmarried','Unmarried','Unmarried','Married'],
           'Class':['No','No','Yes','No','Yes','Yes','No','Yes','Yes','Yes']}           
    from Chapter_02 import DecisionTree_ID3 as ID3
    #Preprocess data set
    df = ID3.preProcess(dataset)
    
    #Lets build the tree
    tree = ID3.buildTree(df)
    
    import pprint
    #print(tree) 
    pprint.pprint(tree)       
    
    #Select test instance 
    inst = df.ix[2]
    
    #Remove its class attribute
    inst.pop('Class')
    
    #Get prediction
    prediction = ID3.predict(inst, tree)
    print("Prediction: %s"%prediction[0])
    
main()