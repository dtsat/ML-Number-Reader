#!/usr/bin/env python3

# Feature extractors
from svm import SVMFeatureExtractor
from three_by_three import ThreeByThreeFeatureExtractor
from kmeans import KMeansFeatureExtractor
from gradient import GradientFeatureExtractor

# Image loader and cross-validater
from importer import Importer

# Support Vector Machine
from sklearn.svm import SVC

# Multilayer Neural Network
from sklearn.neural_network import MLPClassifier as MLP

# Decision Tree
from sklearn.tree import DecisionTreeClassifier as DTC

# K-Nearest-Neighbours
from sklearn.neighbors import KNeighborsClassifier as KNC

# KMeans clusterer
from sklearn.cluster import KMeans

#
from sklearn.metrics import confusion_matrix

import json
import csv

class Classifier:
    """
    This class aggregates the tests we run using 3 descriptors and 4 ML algorithms.
    """

    def __init__(self, numImages=5, crossValidate=False):
        """
        Parameters:
            -- numImages: how many numbers to use in test (default: 5, max: 9, min: 2)
            -- crossValidate: randomly select images for learn, validate, and test sets
        """
        self.__range = numImages
        self.__crossValidate = crossValidate
        self.validateResults = {
            # Descriptor in SVM paper
            'svm': {
                'MLP': 0,
                'DTC': 0,
                'SVC': 0,
                'KNC': 0,   
            },

            # Descriptor in 3x3 paper
            'three': {
                'MLP': 0,
                'DTC': 0,
                'SVC': 0,
                'KNC': 0,   
            },

            # Descriptor from the kmeans paper
            'gradient': {
                'MLP': 0,
                'DTC': 0,
                'SVC': 0,
                'KNC': 0,
            },
        }

        if crossValidate:
            self.testResults = {
                # Descriptor in SVM paper
                'svm': {
                    'MLP': [],
                    'DTC': [],
                    'SVC': [],
                    'KNC': [],   
                },
    
                # Descriptor in 3x3 paper
                'three': {
                    'MLP': [],
                    'DTC': [],
                    'SVC': [],
                    'KNC': [],   
                },
    
                # Descriptor from the kmeans paper
                'gradient': {
                    'MLP': [],
                    'DTC': [],
                    'SVC': [],
                    'KNC': [],
                },
            }

        else:
            self.testResults = {
                # Descriptor in SVM paper
                'svm': {
                    'MLP': 0,
                    'DTC': 0,
                    'SVC': 0,
                    'KNC': 0,   
                },
    
                # Descriptor in 3x3 paper
                'three': {
                    'MLP': 0,
                    'DTC': 0,
                    'SVC': 0,
                    'KNC': 0,   
                },
    
                # Descriptor from the kmeans paper
                'gradient': {
                    'MLP': 0,
                    'DTC': 0,
                    'SVC': 0,
                    'KNC': 0,
                },
            }

    
        # Run tests using 3 methods of feature extraction
    
        if crossValidate:
            for offset in range(0, 1000, 100):
                
                print("Cross-validating at offset:", offset)
                self.load_sets('three', crossValidate, offset)
                self.__learn_and_validate()
        
                self.load_sets('gradient', crossValidate, offset)
                self.__learn_and_validate()
        
                self.load_sets('svm', crossValidate, offset)
                self.__learn_and_validate()
                print("Test Results: ")
                print(json.dumps(self.testResults, indent=4, sort_keys=True))
        
        else: 
            
            print('Loading images sequentially:')
            self.load_sets('three', crossValidate)
            self.__learn_and_validate()
    
            self.load_sets('gradient', crossValidate)
            self.__learn_and_validate()
    
            self.load_sets('svm', crossValidate)
            self.__learn_and_validate()
            print("Results: ")
            print("Validate Results")
            print(json.dumps(self.validateResults, indent=4, sort_keys=True))
            print("Test Results")
            print(json.dumps(self.testResults, indent=4, sort_keys=True))
            
        results = {'validateResults': self.validateResults, 'testResults': self.testResults}
            
        with open('./results.json', 'w') as file:
            json.dump(results, file, indent=4)
            file.close()

    def load_sets(self, which, crossValidate=False, offset=0):
        """
        Parses the images for features and loads the descriptors
        using the specified method of extraction.

        Parameters:
            -- which: `svm`, `kmeans`, or `three`
            -- crossValidate: Set by the constructor
        """
        if which == 'svm':
            self.extractor = SVMFeatureExtractor()
        # if which == 'kmeans':
        #     self.extractor = KMeansFeatureExtractor()
        if which == 'three':
            self.extractor = ThreeByThreeFeatureExtractor()
        if which == 'gradient':
            self.extractor = GradientFeatureExtractor()
        self.__load_descriptors(offset)
        self.__which = which

    def __load_descriptors(self, offset):
        """
        Loads the descriptors. Private method.
        """
        self.learn = [[], []]
        self.validate = [[], []]
        self.test = [[], []]

        # Load images for learn and validate sets
        # TODO: Load and use test set
        for dir in range(self.__range):
            importer = Importer('images/' + str(dir), self.__crossValidate, offset)

            #print("Importing learn from " + str(dir) + " using " + str(self.extractor))
            for image in importer.learn:
                self.learn[0].append(self.extractor.run(image).data)
                self.learn[1].append(dir)

            #print("Importing validate from " + str(dir) + " using " + str(self.extractor))
            for image in importer.validate:
                self.validate[0].append(self.extractor.run(image).data)
                self.validate[1].append(dir)
                
            for image in importer.test:
                self.test[0].append(self.extractor.run(image).data)
                self.test[1].append(dir)

    def __learn_and_validate(self):
        """
        Here comes the magic. Use ML algorithms to learn, validate, and test.
        """
        
        print('Using extractor:', self.extractor, '\n')
        
        ###################################################################
        print("Learning using SVC...")
        classifier = SVC(C=1.0, kernel='sigmoid', gamma='auto', probability=False, verbose=True, max_iter=60000, decision_function_shape='ovr')
        classifier.fit(self.learn[0], self.learn[1])

        print("Validating using SVC...")
        self.validateResults[self.__which]['SVC'] = classifier.score(self.validate[0],self.validate[1])        
        
        print("Testing using SVC...")
        predicted = classifier.predict(self.test[0])
        expected = self.test[1]
        matrix = confusion_matrix(expected, predicted)
        self.writeConfusionMatrixCsv('SVC', matrix)
        print('\n', matrix, '\n')
        
        if self.__crossValidate: 
            self.testResults[self.__which]['SVC'].append(classifier.score(self.test[0], self.test[1]))
        else:
            self.testResults[self.__which]['SVC'] = classifier.score(self.test[0], self.test[1])
        
        ####################################################################
        
        print("Learning using MLP...")
        classifier = MLP(solver='lbfgs', alpha=1e-5, random_state=1)
        classifier.fit(self.learn[0], self.learn[1])

        print("Validating using MLP...")
        self.validateResults[self.__which]['MLP'] = classifier.score(self.validate[0],self.validate[1])
        
        print("Testing using MLP...")
        predicted = classifier.predict(self.test[0])
        expected = self.test[1]
        matrix = confusion_matrix(expected, predicted)
        self.writeConfusionMatrixCsv('MLP', matrix)
        print('\n', matrix, '\n')
        
        if self.__crossValidate: 
            self.testResults[self.__which]['MLP'].append(classifier.score(self.test[0], self.test[1]))
        else:
            self.testResults[self.__which]['MLP'] = classifier.score(self.test[0], self.test[1])
        

        ########################################################################3
        
        
        print("Learning using DTC...")
        classifier = DTC(random_state=0)
        classifier.fit(self.learn[0], self.learn[1])
    
        print("Validating using DTC...")
        self.validateResults[self.__which]['DTC'] = classifier.score(self.validate[0], self.validate[1])

        print("Testing using DTC...")
        predicted = classifier.predict(self.test[0])
        expected = self.test[1]
        matrix = confusion_matrix(expected, predicted)
        self.writeConfusionMatrixCsv('DTC', matrix)
        print('\n', matrix, '\n')
        
        if self.__crossValidate: 
            self.testResults[self.__which]['DTC'].append(classifier.score(self.test[0], self.test[1]))
        else:
            self.testResults[self.__which]['DTC'] = classifier.score(self.test[0], self.test[1])
        
        ##########################################################################
        
        print("Learning using KNC...")
        classifier = KNC(n_neighbors=self.__range-1)
        classifier.fit(self.learn[0], self.learn[1])
        
        print("Validating using KNC...")
        self.validateResults[self.__which]['KNC'] = classifier.score(self.validate[0], self.validate[1])
        
        print("Testing using KNC...")
        predicted = classifier.predict(self.test[0])
        expected = self.test[1]
        matrix = confusion_matrix(expected, predicted)
        self.writeConfusionMatrixCsv('KNC', matrix)
        print('\n', matrix, '\n')
        
        if self.__crossValidate: 
            self.testResults[self.__which]['KNC'].append(classifier.score(self.test[0], self.test[1]))
        else:
            self.testResults[self.__which]['KNC'] = classifier.score(self.test[0], self.test[1])

        
    def writeConfusionMatrixCsv(self, technique, matrix):
        if not self.__crossValidate:
            with open('./confusion_csv/' + str(self.extractor) + '_' + technique + '.csv', 'w') as file:
                writer = csv.writer(file)
                
                for l in matrix:
                    writer.writerow(l)
                
                file.close()
        else:
            pass
        
            
            
