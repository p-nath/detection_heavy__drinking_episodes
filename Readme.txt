## Detection of Heavy Drinking Episodes Using Smartphone Accelerometer Data
Details
---

1. clean_tac: folder containing all the TAC readings for the 13 candidates
2. all_accelerometer_data_pids_13: contains the tri-axial accelerometer data for 13 PIDs
3. final_data.csv: 
 - output of gen_features; contains all the features we extracted using the 2 window approach and summarizing statistics
 - 30874 datapoints x 180 features
 - this data is used by model_train to train the random forest classifier to get an accuracy of 82.27 %
3. gen_features.py: python file to generate features from the raw data
4. model_train: python file to use random forest classifier using different values of max depth ranging from 1 to 20
   We have seen that for max_depth=20 the accuracy is the best so we ran it with a random seed fro 20 times and taken the average accuracy.
   It also shows a plot of how accuracy increases with max_depth.

To run any of the scripts - 
python3 <script.py>

References
---

[Learning to Detect Heavy Drinking Episodes
Using Smartphone Accelerometer Data](http://ceur-ws.org/Vol-2429/paper6.pdf#page=8&zoom=100,72,509)

[dataset](https://archive.ics.uci.edu/ml/datasets/Bar+Crawl%3A+Detecting+Heavy+Drinking)