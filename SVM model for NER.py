import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error

#PART 2
# Load file
df = pd.read_csv(r"C:\Users\gandh\OneDrive\Desktop\test_dataset (1).csv")
df['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)
response = 'Spending Score (1-100)'
predictors = ['Gender', 'Age', 'Annual Income (k$)']

# Initialize variables
i = 1

# Starting and ending index for each fold
for fold in range(10):
    start = int(fold * (len(df) / 10)) #starts at 0, then 20, then 40 -> all the way to 180 making ten folds. This is starting index
    end = int((fold + 1) * (len(df) / 10)) #starts at 20, goes all the way to 200 making 10 folds. This is ending index

    splicedataindexinten = df[start:end]  #This is the data for the fold that we are currently looking at, also known as the testing data
    traindata = pd.concat([df[:start], df[end:]]) # This is anything but the spliced data or testing data, so it is the training data 

    bestPredictors = []  #we will append the best predictors into this
    selPredictors = [] #These are the current predictors we will be working with 
    bestMSE = 100000000000 #MAX value 
    bestPredictor = ''

    for p in predictors:
        if p not in selPredictors:
            selPredictors.append(p)
            xtrain = traindata[selPredictors]
            ytrain = traindata[response]
            #xtrain contains the training data for each of the variables 
            #ytrain contains the response spending score 
            # Create and train the SVM model
            svmmodel = SVC(kernel='linear')
            svmmodel.fit(xtrain, ytrain)
            
            xval = splicedataindexinten[selPredictors]
            yval = splicedataindexinten[response]

            # Predict using the trained SVM model
            ypredicted = svmmodel.predict(xval)
            
            #find the Mean squared error average for each predictor 
            mse = mean_squared_error(yval, ypredicted)
            print(str(mse) + "--->" + p + "MSE")
            
            if mse <= bestMSE:
                bestMSE = mse
                bestPredictor = p
        #finds the smallest MSE value out of each predictor so that it could be appended into the bestpredictor array and also
        #be printed out based on the last data appended which would have the lowest MSE value
        bestPredictors.append(bestPredictor)

    print("At fold " + str(i) + " the best predictor with the lowest MSE value is " +  bestPredictors[len(bestPredictors) - 1] + "\n")
    #This prints out the predictor with the lowest MSE value

    #PART 3:
    # Prompt generation function
    def generate_prompt(Age, Gender, AnnualIncome):
        prompt = "Predict the spending score for a"  + str(Age) + " -year-old " + str(Gender) + "with an  AnnualIncome of $" + AnnualIncome
        return prompt

    # User input
    Age = input("Enter Age: ")
    Gender = input("Enter Gender (Male/Female): ")
    AnnualIncome = input("Enter annual AnnualIncome: ")

    # Generate prompt
    prompt = generate_prompt(Age, Gender, AnnualIncome)

    # Simulated language model response
    simresponse = "MSE between actual vs predicted spending score:" + str(bestMSE)

    print("\nLanguage Model Response:")
    print(simresponse)
    i = i + 1

