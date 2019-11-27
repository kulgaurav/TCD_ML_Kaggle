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

    data = pd.read_csv(
        'tcd-ml-1920-group-income-train.csv')
    data.sort_values("Instance", inplace=True)
    data.drop_duplicates(subset="Instance",
                                     keep=False, inplace=True)

    test_data = pd.read_csv(
        'tcd-ml-1920-group-income-test.csv')
 
    data = data.loc[:, [
        'Year of Record', 'Housing Situation', 'Crime Level in the City of Employement', 'Work Experience in Current Job [years]', 'Satisfation with employer', 'Gender', 'Age', 'Country', 'Size of City', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Body Height [cm]', 'Yearly Income in addition to Salary (e.g. Rental Income)', 'Total Yearly Income [EUR]', ]]
    test_data = test_data.loc[:, ['Year of Record', 'Housing Situation', 'Crime Level in the City of Employement', 'Work Experience in Current Job [years]', 'Satisfation with employer', 'Gender', 'Age',
                                        'Country', 'Size of City', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Body Height [cm]', 'Yearly Income in addition to Salary (e.g. Rental Income)', 'Total Yearly Income [EUR]', ]]

    data = parse_eur(data)
    test_data = parse_eur(test_data)

    data['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(
        data['Yearly Income in addition to Salary (e.g. Rental Income)'], errors='coerce')
    data['Work Experience in Current Job [years]'] = pd.to_numeric(
        data['Work Experience in Current Job [years]'], errors='coerce')
    data = data.replace(np.nan, 0, regex=True)


    
    test_data['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(
        test_data['Yearly Income in addition to Salary (e.g. Rental Income)'], errors='coerce')
    test_data['Work Experience in Current Job [years]'] = pd.to_numeric(
        test_data['Work Experience in Current Job [years]'], errors='coerce')
    test_data = test_data.replace(np.nan, 0, regex=True)


    data[["Year of Record"]] = data[["Year of Record"]].fillna(value=int(data["Year of Record"].mean()))


    test_data[["Year of Record"]] = test_data[["Year of Record"]].fillna(value=int(test_data["Year of Record"].mean()))

    data[['Hair Color']] = data[["Hair Color"]].fillna(value="unknownHC")
    data[['Hair Color']] = data[['Hair Color']].replace('0', 'unknownHC') 
    data[['Hair Color']] = data[['Hair Color']].replace('Unknown', 'unknownHC')
    data[['Hair Color']] = data[['Hair Color']].replace('nan', 'unknownHC')


    data[['Housing Situation']] = data[['Housing Situation']].fillna(value="unknownHC")
    data[['Housing Situation']] = data[['Housing Situation']].replace('0', 'unknownHS') 
    data[['Housing Situation']] = data[['Housing Situation']].replace(0, 'unknownHS')
    data[['Housing Situation']] = data[['Housing Situation']].replace('nA', 'unknownHS')


    data[["Gender"]] = data[["Gender"]].fillna(value="unknownG")
    data[['Gender']] = data[['Gender']].replace('0', 'unknownG') 
    data[['Gender']] = data[['Gender']].replace('unknown', 'unknownG')
    data[['Gender']] = data[['Gender']].replace('f', 'female')

    data[["University Degree"]] = data[["University Degree"]].fillna(data["University Degree"].mode()[0])
    

    data[["Profession"]] = data[["Profession"]].fillna(value="unknownP")


    data[['Satisfation with employer']] = data[['Satisfation with employer']].replace(0, 'unknown')
    data[['Satisfation with employer']] = data[['Satisfation with employer']].fillna('unknown')
    data[['Gender']] = data[['Gender']].replace(0, 'unknownG') 
    test_data[['Satisfation with employer']] = test_data[['Satisfation with employer']].replace(0, 'unknown')
    test_data[['Satisfation with employer']] = test_data[['Satisfation with employer']].fillna('unknown')
    test_data[['Gender']] = test_data[['Gender']].replace(0, 'unknownG') 


    test_data[["Hair Color"]] = test_data[["Hair Color"]].fillna(value="unknownHC")
    test_data[['Hair Color']] = test_data[['Hair Color']].replace('0', 'unknownHC') 
    test_data[['Hair Color']] = test_data[['Hair Color']].replace('Unknown', 'unknownHC')
    test_data[['Hair Color']] = test_data[['Hair Color']].replace('nan', 'unknownHC')

    test_data[["Housing Situation"]] = test_data[['Housing Situation']].fillna(value="unknownHC")
    test_data[['Housing Situation']] = test_data[['Housing Situation']].replace('0', 'unknownHS') 
    test_data[['Housing Situation']] = test_data[['Housing Situation']].replace(0, 'unknownHS')
    test_data[['Housing Situation']] = test_data[['Housing Situation']].replace('nA', 'unknownHS')


    test_data[["University Degree"]] = test_data[["University Degree"]].fillna(test_data["University Degree"].mode()[0])

    test_data[["Profession"]] = test_data[["Profession"]].fillna(value="unknownP")

    test_data[["Gender"]] = test_data[["Gender"]].fillna(value="unknownG")
    test_data[['Gender']] = test_data[['Gender']].replace('0', 'unknownG') 
    test_data[['Gender']] = test_data[['Gender']].replace('unknown', 'unknownG')
    test_data[['Gender']] = test_data[['Gender']].replace('f', 'female')

    
    test_data[["Country"]] = test_data[["Country"]].fillna(test_data["Country"].mode()[0])


    groupedProf = data.groupby('Profession', as_index=False)['Total Yearly Income [EUR]'].mean()
    data['Profession'] = data['Profession'].map(groupedProf.set_index('Profession')['Total Yearly Income [EUR]'])

    groupedCountry = data.groupby('Country', as_index=False)['Total Yearly Income [EUR]'].mean()
    data['Country'] = data['Country'].map(groupedCountry.set_index('Country')['Total Yearly Income [EUR]'])

    groupedUD = data.groupby('University Degree', as_index=False)['Total Yearly Income [EUR]'].mean()
    data['University Degree'] = data['University Degree'].map(groupedUD.set_index('University Degree')['Total Yearly Income [EUR]'])

    groupedG = data.groupby('Gender', as_index=False)['Total Yearly Income [EUR]'].mean()
    data['Gender'] = data['Gender'].map(groupedG.set_index('Gender')['Total Yearly Income [EUR]'])

    groupedHS = data.groupby('Housing Situation', as_index=False)['Total Yearly Income [EUR]'].mean()
    data['Housing Situation'] = data['Housing Situation'].map(groupedHS.set_index('Housing Situation')['Total Yearly Income [EUR]'])

    groupedWE = data.groupby('Work Experience in Current Job [years]', as_index=False)['Total Yearly Income [EUR]'].mean()
    data['Work Experience in Current Job [years]'] = data['Work Experience in Current Job [years]'].map(groupedWE.set_index('Work Experience in Current Job [years]')['Total Yearly Income [EUR]'])

    groupedSWE = data.groupby('Satisfation with employer', as_index=False)['Total Yearly Income [EUR]'].mean()
    data['Satisfation with employer'] = data['Satisfation with employer'].map(groupedSWE.set_index('Satisfation with employer')['Total Yearly Income [EUR]'])

    #groupedYIA = data.groupby('Yearly Total Yearly Income [EUR] in addition to Salary (e.g. Rental Total Yearly Income [EUR])', as_index=False)['Total Yearly Income [EUR]'].mean()
    #data['Yearly Total Yearly Income [EUR] in addition to Salary (e.g. Rental Total Yearly Income [EUR])'] = data['Yearly Total Yearly Income [EUR] in addition to Salary (e.g. Rental Total Yearly Income [EUR])'].map(groupedYIA.set_index('Yearly Total Yearly Income [EUR] in addition to Salary (e.g. Rental Total Yearly Income [EUR])')['Total Yearly Income [EUR]'])

    groupedHC = data.groupby('Hair Color', as_index=False)['Total Yearly Income [EUR]'].mean()
    data['Hair Color'] = data['Hair Color'].map(groupedHC.set_index('Hair Color')['Total Yearly Income [EUR]'])

    test_data['Profession'] = test_data['Profession'].map(groupedProf.set_index('Profession')['Total Yearly Income [EUR]'])
    test_data['Country'] = test_data['Country'].map(groupedCountry.set_index('Country')['Total Yearly Income [EUR]'])
    test_data['University Degree'] = test_data['University Degree'].map(groupedUD.set_index('University Degree')['Total Yearly Income [EUR]'])
    test_data['Gender'] = test_data['Gender'].map(groupedG.set_index('Gender')['Total Yearly Income [EUR]'])
    test_data['Housing Situation'] = test_data['Housing Situation'].map(groupedHS.set_index('Housing Situation')['Total Yearly Income [EUR]'])
    test_data['Work Experience in Current Job [years]'] = test_data['Work Experience in Current Job [years]'].map(groupedWE.set_index('Work Experience in Current Job [years]')['Total Yearly Income [EUR]'])
    test_data['Satisfation with employer'] = test_data['Satisfation with employer'].map(groupedSWE.set_index('Satisfation with employer')['Total Yearly Income [EUR]'])
    test_data['Hair Color'] = test_data['Hair Color'].map(groupedHC.set_index('Hair Color')['Total Yearly Income [EUR]'])

    

    y = data['Total Yearly Income [EUR]'].apply(np.log)
    data = data.drop("Total Yearly Income [EUR]", 1)
    test_data = test_data.drop("Total Yearly Income [EUR]", 1)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('target', TargetEncoder())])


    numeric_features = data.select_dtypes(
        include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(
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

    X = data
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

    reg.fit(data, y.values)
    pred_test = reg.predict(test_data)
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
