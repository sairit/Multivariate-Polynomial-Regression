"""
@Author: Sai Yadavalli
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class PolyRegression:
    def __init__ (self, featureA, featureB, degree, to_train=False):
        """
        Initializes the polynomial regression model.
        
        Args:
            featureA (str): Name of the independent variable.
            featureB (str): Name of the dependent variable.
            degree (int): Degree of the polynomial regression.
            to_train (bool, optional): Whether to train the model upon initialization. Defaults to False.
        """

        self.W = None
        self.degree = degree
        self.data = None
        self.X = None
        self.Y = None
        self.train = pd.DataFrame()

        # Only train model if specified
        if to_train:
            self.split_data()
            self.training(featureA, featureB, degree)

    def set_weights(self, w):
        """
        Presets weights

        Args:
            w (list): List of floats representing weights.
        """
        self.W = w

    def load_data(self):
        """
        Loads data from a CSV file specified by the user.

        Returns:
            pd.DataFrame: Loaded dataset.
        """

        # Ask user for file name
        file_path = input("Enter the .csv data file name: ")
        file_path = f"Data/{file_path}.csv"
        
        # Read data and print column names
        df = pd.read_csv(file_path)
        print("The data headers are: ", list(df))

        self.data = df
        return df
    
    # Method to split data into training and testing sets
    def split_data(self):
        """
        Splits the dataset into five folds for cross-validation and saves them as CSV files.
        """

        if self.data is None:
            self.load_data()

        if not os.path.exists("PolyRegression/train"):
            os.makedirs("PolyRegression/train")

        df = self.data

        # Split data into 5 folds
        f1 = df.sample(frac=0.2)
        df2 = df.drop(f1.index)
        f2 = df2.sample(frac=0.25)
        df3 = df2.drop(f2.index)
        f3 = df3.sample(frac=(1/3))
        df4 = df3.drop(f3.index)
        f4 = df4.sample(frac=0.5)
        f5 = df4.drop(f4.index)

        # Concatenate the folds to create training and testing sets
        train1 = pd.concat([f2,f3,f4,f5])
        train2 = pd.concat([f1,f3,f4,f5])
        train3 = pd.concat([f1,f2,f4,f5])
        train4 = pd.concat([f1,f2,f3,f5])
        train5 = pd.concat([f1,f2,f3,f4])

        # Save test and validation sets as separate files
        train1.to_csv("PolyRegression/train/train1.csv", index = False)
        f1.to_csv("PolyRegression/train/validation1.csv", index = False)
        train2.to_csv("PolyRegression/train/train2.csv", index = False)
        f2.to_csv("PolyRegression/train/validation2.csv", index = False)
        train3.to_csv("PolyRegression/train/train3.csv", index = False)
        f3.to_csv("PolyRegression/train/validation3.csv", index = False)
        train4.to_csv("PolyRegression/train/train4.csv", index = False)
        f4.to_csv("PolyRegression/train/validation4.csv", index = False)
        train5.to_csv("PolyRegression/train/train5.csv", index = False)
        f5.to_csv("PolyRegression/train/validation5.csv", index = False)


    # Matrify the data and create the X and Y matrices
    def matrix(self, trainFeature, testFeature, train):
        """
        Converts the data into NumPy matrices X and Y.
        
        Args:
            trainFeature (str): Name of the independent variable.
            testFeature (str): Name of the dependent variable.
            train (pd.DataFrame): Training dataset.
        
        Returns:
            tuple: X and Y matrices as NumPy arrays.
        """

        # Create X matrix
        x = train[[trainFeature]].to_numpy()

        # Create Y matrix
        y = train[[testFeature]].to_numpy()

        return x, y

    # Method to calculate training and testing costs along with the weights
    def costs(self, X, Y, w=None):
        """
        Computes the cost function and weights for polynomial regression.
        
        Args:
            X (np.array): Feature matrix.
            Y (np.array): Target values.
            w (np.array, optional): Precomputed weights. Defaults to None.
        
        Returns:
            tuple: Computed cost and weight matrices.
        """

        # Calculate weights
        W = w
        m = len(Y)
        A = np.linalg.pinv(np.dot(X.T, X))
        B = np.dot(X.T, Y)
        if w is None:
            W = np.dot(A, B)

        # Caclulate cost
        A = np.dot(X, W) - Y
        J = (1/m) * np.dot(A.T, A)

        return J, W

    # Plot hypothesis function and training data
    def plot_regression(self, featureA, featureB):
        """
        Plots the regression curve along with the training data.
        
        Args:
            featureA (str): Name of the independent variable.
            featureB (str): Name of the dependent variable.
        """

        # Plot the training data
        plt.figure()
        plt.scatter(self.X, self.Y, color="blue", label="Training Data")
        
        # Plot each regression
        for i in range(1, self.degree + 1):  
            regression_curve = self.calculate_regression(self.X, self.Y, i)
            plt.plot(self.X, regression_curve, label=f"Degree {i} Regression Curve")


        plt.title(f"{featureA} vs. {featureB}")
        plt.xlabel(featureA)
        plt.ylabel(featureB)
        plt.legend(loc="best")
        plt.show()

    def polynomial_features(self, X, degree):
        """
        Generates polynomial features for a given input matrix.
        
        Args:
            X (np.array): Input feature matrix.
            degree (int): Polynomial degree.
        
        Returns:
            np.array: Matrix with polynomial features.
        """

        x_poly = np.ones((X.shape[0], 1))  

        for d in range(1, degree + 1):
            x_poly = np.hstack((x_poly, X ** d)) 

        return x_poly

    # Find the training and testing costs for each model
    def training(self, featureA, featureB, degree):
        """
        Trains the polynomial regression model using cross-validation.
        
        Args:
            featureA (str): Name of the independent variable.
            featureB (str): Name of the dependent variable.
            degree (int): Polynomial degree.
        """

        # Create X and Y matrices for overall data set
        self.X, self.Y = self.matrix(featureA, featureB, self.data)
        
        # Perform training up until desired polynomial degree
        for i in range(1, degree+1):
            costs = np.empty((0,))
            set = np.empty((0,), dtype=object)
            fold_weights = np.empty((0, i+1))

            # Train on each fold
            for j in range(1, 6):
                train = pd.read_csv(f"PolyRegression/train/train{j}.csv")
                validation = pd.read_csv(f"PolyRegression/train/validation{j}.csv")

                # Create X and Y matrices for train set
                x, y = self.matrix(train.columns[0], train.columns[1], train)
                
                # Compute training matrix for polynomial features
                x = self.polynomial_features(x, i)
                
                # Calculate training weights and cost
                j, w = self.costs(x, y)
                fold_weights = np.vstack((fold_weights, w.T))
                costs = np.append(costs, j.flatten(), axis=0)
                set = np.append(set, np.array([f"Train {int(j)}"]), axis=0)

                # Create X and Y matrices for validation sets
                x, y = self.matrix(validation.columns[0], validation.columns[1], validation)
                
                # Compute validation matrix for polynomial features                
                x = self.polynomial_features(x, i)

                # Find cost for validation based on weights found from training
                j, w = self.costs(x, y, w)
                costs = np.append(costs, j.flatten(), axis=0)
                set = np.append(set, np.array([f"Validation {int(j)}"]), axis=0)

            self.train["Set"] = set
            self.train[f"Degree {i}"] = costs
        
    def calculate_regression(self, x, y, degree):
        """
        Produces predictions from the polynomial regression model.
        
        Args:
            x (numpy matrix): Matrix for training feature data.
            y (numpy matrix): Matrix for predicting feature data.
            degree (int): Polynomial degree.
        """

        x = self.polynomial_features(x, degree)
        j, w = self.costs(x, y)
        regression = np.dot(x, w)

        print(f"Polynomial Degree {degree} Weights: ", w)
        return regression
    
    def test(self):
        """
        Predicts the value for a given input based on trained weights.
        """

        w = self.W
        year = int(input("Enter number of year(s) after 1900: ")) 
        year_array = np.array([[year]])
        x_test = self.polynomial_features(year_array, len(w) - 1)
        w = np.array(w).reshape(-1, 1)  
        prediction = np.dot(x_test, w)[0, 0]

        print(f"The predicted average temperature for {year}(s) after 1900 is: {prediction:.2f}°F")


    def train_table(self):
        """
        Prints and visualizes training and validation costs for different polynomial degrees.
        """

        print(self.train)
        degrees = range(1, self.degree + 1)

        for i in degrees:
            costs = self.train[f"Degree {i}"]
            train_costs = costs[0::2]  
            validation_costs = costs[1::2] 

            plt.figure(figsize=(8, 5))
            plt.plot(np.array(range(1, 6)), train_costs, label="Train Costs")
            plt.plot(np.array(range(1, 6)), validation_costs, label="Validation Costs")

            plt.title(f"Training and Validation Costs for Degree {i}")
            plt.xlabel("Fold Index")
            plt.ylabel("Cost")
            plt.legend(loc="best")

            plt.show()

            print(f"Degree {i} Average Train Costs: {np.mean(train_costs)}")
            print(f"Degree {i} Average Validation Cost: {np.mean(validation_costs)}")

# MAIN

# Change to true and update features to train model on new data
model = PolyRegression("Year","Avg_Tmp", 3, False)
model.set_weights([5.27096769e+01, -8.51186716e-02, 1.42542178e-03, -5.18039187e-06])

# Testing model with preset weights
while(input("Do you want to continue with testing? (yes/no): ") == "yes"):
    model.test()



    

