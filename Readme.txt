MACHINE LEARNING NUMBER READER
David Tsatsoulis
Vanessa Kurt
Quentin Dumoulin
Lev Kokotov
David Desmarais-Michaud

This report and project details the implementation and results of several classification and clustering Machine Learning methods used to analyze a database of handwritten Persian digits. 

The database contains 10,000 images of black and white handwritten digits, ranging from 0 to 9(equally weighted). 

The data is divided into training, validation and test sets. A number of data extraction methods are used, including Gradient, Derivative and 3 x 3 methods taken from another paper. 

Classification is done using a Support Vector Classifier, a K-Nearest-Neighbor Classifier, a Multi-Layer Perceptron classifier, and a Decision Tree Classifier. 
Clustering is done using the K-means, Affinity Propagation and Birch methods. 

All implementations are done in Python using the scikit-learn library. 

Results were evaluated and validated using Cross-Validation and Confusion Matrices. 

The results of the experiments conducted in the project favour the use of the 3 x 3 data extraction method, coupled with the K-Nearest-Neighbor classifier and Affinity Propagation cluster method. 
The Gradient method of extraction and the Derivative method of extraction were found to be less accurate for some experiments and produced results that averaged between accurate and random guessing.