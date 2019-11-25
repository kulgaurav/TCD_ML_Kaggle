from sklearn.metrics import mean_absolute_error
import csv
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import category_encoders as ce
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

def main():
    # Note: This program is non-deterministic as it uses RandomForestRegressor
    # If you wish to make the program deterministic set random_state parameter in
    # RandomForestRegressor to a specific integer

    # Importing the dataset
    training_dataset = pd.read_csv(
        'tcd-ml-1920-group-income-train.csv')
    training_dataset.sort_values("Instance", inplace=True)

    # dropping ALL duplicte values
    training_dataset.drop_duplicates(subset="Instance",
                                     keep=False, inplace=True)

    test_dataset = pd.read_csv(
        'tcd-ml-1920-group-income-test.csv')

 
    training_dataset = training_dataset.loc[:, [
        'Year of Record', 'Housing Situation', 'Crime Level in the City of Employement', 'Work Experience in Current Job [years]', 'Satisfation with employer', 'Gender', 'Age', 'Country', 'Size of City', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Body Height [cm]', 'Yearly Income in addition to Salary (e.g. Rental Income)', 'Total Yearly Income [EUR]', ]]
    test_dataset = test_dataset.loc[:, ['Year of Record', 'Housing Situation', 'Crime Level in the City of Employement', 'Work Experience in Current Job [years]', 'Satisfation with employer', 'Gender', 'Age',
                                        'Country', 'Size of City', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Body Height [cm]', 'Yearly Income in addition to Salary (e.g. Rental Income)', 'Total Yearly Income [EUR]', ]]

    training_dataset = parse_eur(training_dataset)
    test_dataset = parse_eur(test_dataset)

    training_dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(
        training_dataset['Yearly Income in addition to Salary (e.g. Rental Income)'], errors='coerce')
    training_dataset['Work Experience in Current Job [years]'] = pd.to_numeric(
        training_dataset['Work Experience in Current Job [years]'], errors='coerce')
    training_dataset = training_dataset.replace(np.nan, 0, regex=True)
    
    test_dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(
        test_dataset['Yearly Income in addition to Salary (e.g. Rental Income)'], errors='coerce')
    test_dataset['Work Experience in Current Job [years]'] = pd.to_numeric(
        test_dataset['Work Experience in Current Job [years]'], errors='coerce')
    test_dataset = test_dataset.replace(np.nan, 0, regex=True)

    training_dataset = training_dataset.drop('Wears Glasses', axis=1)
    test_dataset = test_dataset.drop('Wears Glasses', axis=1)

    training_dataset[["Year of Record"]] = training_dataset[["Year of Record"]].fillna(value=training_dataset["Year of Record"].mode()[0])

    test_dataset[["Year of Record"]] = test_dataset[["Year of Record"]].fillna(value=test_dataset["Year of Record"].mode()[0])

    training_dataset[['Hair Color']] = training_dataset[["Hair Color"]].fillna(value="unknownHC")
    training_dataset[['Hair Color']] = training_dataset[['Hair Color']].replace('0', 'unknownHC') 
    training_dataset[['Hair Color']] = training_dataset[['Hair Color']].replace('Unknown', 'unknownHC')
    training_dataset[['Hair Color']] = training_dataset[['Hair Color']].replace('nan', 'unknownHC')


    training_dataset[['Housing Situation']] = training_dataset[['Housing Situation']].fillna(value="unknownHC")
    training_dataset[['Housing Situation']] = training_dataset[['Housing Situation']].replace('0', 'unknownHS') 
    training_dataset[['Housing Situation']] = training_dataset[['Housing Situation']].replace(0, 'unknownHS')
    training_dataset[['Housing Situation']] = training_dataset[['Housing Situation']].replace('nA', 'unknownHS')


    training_dataset[["Gender"]] = training_dataset[["Gender"]].fillna(value="unknownG")
    training_dataset[['Gender']] = training_dataset[['Gender']].replace('0', 'unknownG') 
    training_dataset[['Gender']] = training_dataset[['Gender']].replace('unknown', 'unknownG')
    training_dataset[['Gender']] = training_dataset[['Gender']].replace('f', 'female')

    training_dataset[["University Degree"]] = training_dataset[["University Degree"]].fillna(training_dataset["University Degree"].mode()[0])
    

    training_dataset[["Profession"]] = training_dataset[["Profession"]].fillna(value="unknownP")


    training_dataset[['Satisfation with employer']] = training_dataset[['Satisfation with employer']].replace(0, 'unknown')
    training_dataset[['Satisfation with employer']] = training_dataset[['Satisfation with employer']].fillna('unknown')
    training_dataset[['Gender']] = training_dataset[['Gender']].replace(0, 'unknownG') 
    test_dataset[['Satisfation with employer']] = test_dataset[['Satisfation with employer']].replace(0, 'unknown')
    test_dataset[['Satisfation with employer']] = test_dataset[['Satisfation with employer']].fillna('unknown')
    test_dataset[['Gender']] = test_dataset[['Gender']].replace(0, 'unknownG') 


    test_dataset[["Hair Color"]] = test_dataset[["Hair Color"]].fillna(value="unknownHC")
    test_dataset[['Hair Color']] = test_dataset[['Hair Color']].replace('0', 'unknownHC') 
    test_dataset[['Hair Color']] = test_dataset[['Hair Color']].replace('Unknown', 'unknownHC')
    test_dataset[['Hair Color']] = test_dataset[['Hair Color']].replace('nan', 'unknownHC')

    test_dataset[["Housing Situation"]] = test_dataset[['Housing Situation']].fillna(value="unknownHC")
    test_dataset[['Housing Situation']] = test_dataset[['Housing Situation']].replace('0', 'unknownHS') 
    test_dataset[['Housing Situation']] = test_dataset[['Housing Situation']].replace(0, 'unknownHS')
    test_dataset[['Housing Situation']] = test_dataset[['Housing Situation']].replace('nA', 'unknownHS')


    test_dataset[["University Degree"]] = test_dataset[["University Degree"]].fillna(test_dataset["University Degree"].mode()[0])

    test_dataset[["Profession"]] = test_dataset[["Profession"]].fillna(value="unknownP")

    test_dataset[["Gender"]] = test_dataset[["Gender"]].fillna(value="unknownG")
    test_dataset[['Gender']] = test_dataset[['Gender']].replace('0', 'unknownG') 
    test_dataset[['Gender']] = test_dataset[['Gender']].replace('unknown', 'unknownG')
    test_dataset[['Gender']] = test_dataset[['Gender']].replace('f', 'female')

    
    test_dataset[["Country"]] = test_dataset[["Country"]].fillna(test_dataset["Country"].mode()[0])


    groupedProf = training_dataset.groupby('Profession', as_index=False)['Total Yearly Income [EUR]'].mean()
    training_dataset['Profession'] = training_dataset['Profession'].map(groupedProf.set_index('Profession')['Total Yearly Income [EUR]'])

    groupedCountry = training_dataset.groupby('Country', as_index=False)['Total Yearly Income [EUR]'].mean()
    training_dataset['Country'] = training_dataset['Country'].map(groupedCountry.set_index('Country')['Total Yearly Income [EUR]'])

    groupedUD = training_dataset.groupby('University Degree', as_index=False)['Total Yearly Income [EUR]'].mean()
    training_dataset['University Degree'] = training_dataset['University Degree'].map(groupedUD.set_index('University Degree')['Total Yearly Income [EUR]'])

    groupedG = training_dataset.groupby('Gender', as_index=False)['Total Yearly Income [EUR]'].mean()
    training_dataset['Gender'] = training_dataset['Gender'].map(groupedG.set_index('Gender')['Total Yearly Income [EUR]'])

    groupedHS = training_dataset.groupby('Housing Situation', as_index=False)['Total Yearly Income [EUR]'].mean()
    training_dataset['Housing Situation'] = training_dataset['Housing Situation'].map(groupedHS.set_index('Housing Situation')['Total Yearly Income [EUR]'])

    groupedWE = training_dataset.groupby('Work Experience in Current Job [years]', as_index=False)['Total Yearly Income [EUR]'].mean()
    training_dataset['Work Experience in Current Job [years]'] = training_dataset['Work Experience in Current Job [years]'].map(groupedWE.set_index('Work Experience in Current Job [years]')['Total Yearly Income [EUR]'])

    groupedSWE = training_dataset.groupby('Satisfation with employer', as_index=False)['Total Yearly Income [EUR]'].mean()
    training_dataset['Satisfation with employer'] = training_dataset['Satisfation with employer'].map(groupedSWE.set_index('Satisfation with employer')['Total Yearly Income [EUR]'])

    #groupedYIA = training_dataset.groupby('Yearly Total Yearly Income [EUR] in addition to Salary (e.g. Rental Total Yearly Income [EUR])', as_index=False)['Total Yearly Income [EUR]'].mean()
    #training_dataset['Yearly Total Yearly Income [EUR] in addition to Salary (e.g. Rental Total Yearly Income [EUR])'] = training_dataset['Yearly Total Yearly Income [EUR] in addition to Salary (e.g. Rental Total Yearly Income [EUR])'].map(groupedYIA.set_index('Yearly Total Yearly Income [EUR] in addition to Salary (e.g. Rental Total Yearly Income [EUR])')['Total Yearly Income [EUR]'])

    groupedHC = training_dataset.groupby('Hair Color', as_index=False)['Total Yearly Income [EUR]'].mean()
    training_dataset['Hair Color'] = training_dataset['Hair Color'].map(groupedHC.set_index('Hair Color')['Total Yearly Income [EUR]'])

    test_dataset['Profession'] = test_dataset['Profession'].map(groupedProf.set_index('Profession')['Total Yearly Income [EUR]'])
    test_dataset['Country'] = test_dataset['Country'].map(groupedCountry.set_index('Country')['Total Yearly Income [EUR]'])
    test_dataset['University Degree'] = test_dataset['University Degree'].map(groupedUD.set_index('University Degree')['Total Yearly Income [EUR]'])
    test_dataset['Gender'] = test_dataset['Gender'].map(groupedG.set_index('Gender')['Total Yearly Income [EUR]'])
    test_dataset['Housing Situation'] = test_dataset['Housing Situation'].map(groupedHS.set_index('Housing Situation')['Total Yearly Income [EUR]'])
    test_dataset['Work Experience in Current Job [years]'] = test_dataset['Work Experience in Current Job [years]'].map(groupedWE.set_index('Work Experience in Current Job [years]')['Total Yearly Income [EUR]'])
    test_dataset['Satisfation with employer'] = test_dataset['Satisfation with employer'].map(groupedSWE.set_index('Satisfation with employer')['Total Yearly Income [EUR]'])
    test_dataset['Hair Color'] = test_dataset['Hair Color'].map(groupedHC.set_index('Hair Color')['Total Yearly Income [EUR]'])

    

    y = training_dataset['Total Yearly Income [EUR]'].apply(np.log)
    training_dataset = training_dataset.drop("Total Yearly Income [EUR]", 1)
    test_dataset = test_dataset.drop("Total Yearly Income [EUR]", 1)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('target', TargetEncoder())])


    numeric_features = training_dataset.select_dtypes(
        include=['int64', 'float64']).columns
    categorical_features = training_dataset.select_dtypes(
        include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    reg = Pipeline(steps=[('preprocessor', preprocessor),
                          #('regressor', RandomForestRegressor(n_estimators=50))])
                          ('regressor', CatBoostRegressor(iterations=3000,
                             learning_rate=0.1,
                             depth=10,
                             eval_metric='MAE',
                             random_seed = 23,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100))])

                            




    X = training_dataset
    Y = y.values
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)

    # reg.fit(X_train, Y_train,
    #         regressor__eval_set=(X_test,Y_test),
    #         regressor__use_best_model=True,
    #         regressor__verbose=True)
    #pre_test_lgb = reg.predict(X_test)

    #val_mae = mean_absolute_error(Y_test, pre_test_lgb)
    #print(val_mae)

    # Kaggle Test
    reg.fit(training_dataset, y.values)
    pred_test = reg.predict(test_dataset)
    pred_test = np.exp(pred_test)
    
    writeAnswer(pred_test)


# Prints and returns the mean_squared_error
def getScore(regr, test_y, y_pred):
    # Mean squared error
    mse = np.sqrt(mean_squared_error(test_y, y_pred))
    print("Root Mean squared error: %.2f"
          % mse)
    return mse

# Function that writes predictions to file


def writeAnswer(y_pred):
    i = 1
    res = 'Instance,Total Yearly Income [EUR]\n'
    for x in y_pred:
        res = res + str(i) + ',' + str(x) + '\n'
        i = i + 1
    with open('tcd-ml-1920-group-income-submission.csv', 'w') as f:
        f.writelines(res)


def parse_eur(data):
    column = data['Yearly Income in addition to Salary (e.g. Rental Income)']
    temp = [s.replace(' EUR', '') for s in column]
    data['Yearly Income in addition to Salary (e.g. Rental Income)'] = temp
    return data


if __name__ == "__main__":
    main()
