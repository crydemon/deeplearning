import numpy as np
import pandas as pd

if __name__ == '__main__':
    dataset = pd.read_csv('Data.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 3].values
    print("Step 2: Importing dataset")
    print("X")
    print(X)
    print("Y")
    print(Y)
    # Step 3: Handling the missing data
    from sklearn.impute import SimpleImputer as Imputer

    imputer = Imputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    print("---------------------")
    print("Step 3: Handling the missing data")
    print("step2")
    print("X")
    print(X)
    # Step 4: Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    # Creating a dummy variable
    onehotencoder = OneHotEncoder(categorical_features=[0])
    onehotencoder = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [0])],
        # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        remainder='passthrough'  # Leave the rest of the columns untouched
    )
    X = onehotencoder.fit_transform(X)
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)
    print("---------------------")
    print("Step 4: Encoding categorical data")
    print("X")
    print(X)
    print("Y")
    print(Y)

    # Step 5: Splitting the datasets into training sets and Test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print("---------------------")
    print("Step 5: Splitting the datasets into training sets and Test sets")
    print("X_train")
    print(X_train)
    print("X_test")
    print(X_test)
    print("Y_train")
    print(Y_train)
    print("Y_test")
    print(Y_test)

    # Step 6: Feature Scaling
    from sklearn.preprocessing import StandardScaler

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    print("---------------------")
    print("Step 6: Feature Scaling")
    print("X_train")
    print(X_train)
    print("X_test")
    print(X_test)
