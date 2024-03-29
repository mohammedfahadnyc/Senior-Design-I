
I am giving u a slide its outline and what I wanna tell the audience  for each slide
for each slide, i will give u what’s on the slide and then give u info on that slide i wanna tell my audience… 


slide 1 : innovation idea
Service Outage Detection Based On Slack TechCare Chat Data
* Text classification modeling for detecting the likelihood of a service outage

what I wanna tell : Text classification modeling for detecting service outages from Slack TechCare chat data.Explain general approachWe used Slack TechCare chat data train various ML models on the likelihood of a mention of an outage from input text
End idea is to implement a rule set from this predicted outage likelihood to determine if an outage of a service has taken place over a predetermined time period

slide 2 : data collection
*  Colleague 360's TechCare chat data 
* Accessed via Hive Terminal 
* Sample of 15 days chat data, ~30k sample records 

what I wanna tell : Colleague 360’s cl360_luke_conversation Cornerstone table that has TechCare chat data, which queried using our enterprise tool Hive Terminalwe both respectively worked with 15 total days of chat data - sufficient size


slide 3 : Data Processing
Manual Labeling : 	•	Classes: 
* 0 – outage unlikely 
* 1 – outage possible 
* 2 – outage likely 
* ~2.3k unique manually labeled samples 

Semi Supervised Approach
* Training of preliminary model for generating additional pseudo-labels 
* Two language models and two classic algorithms 

Class Augmentation
* Ran semi-supervised pseudo-labeling experiments with and without class augmentation 
* Improvements observed with augmentation of class 2 

what I wanna tell for this slide : We manually labeled a small sample set of the 15 day dataset, ~2300 unique examples, to be used for training a labeling model
* 		Semi-supervised learning approach - idea is to use this preliminary model to supplement our total count of labeled data by generating labels for the rest of the 15-day dataset
* 		This is for time saving purposes, in an ideal scenario we would likely take more time to manually label data
* 		Explain data classes - 0, 1, 2 - show examples
* 		I experimented with class augmentation at this point
* 		Saw an increase in performance with our best labeling models when I increased the size of the 2 class, which had the fewest examples - this class is most important for detecting positive outage cases since it represents outage likely



slide 4 : example of training data


slide 5 : Modeling
2 Language Models (BERT and BART)+ 2 Classic Algorithms(Random forest & Incremental learning classifiers(perception and passive aggressive)
what I wanna tell : We worked with classic algorithms such as random forest and classifiers such as perceptron to take advantage of incremental learning and we also used language models like BERT and BART
* 		Explain that we used language models available from the HuggingFace platform - final models used were Tensorflow-based versions of the language models BERT and BART as these showed the best performance in the pseudo labeling exercise, XLNet and RoBERTa also used for labeling exercise
* 		Explain that these models were used for the pseudo labeling, and then the best performers from that exercise where then trained on the complete 15-day labeled dataset

slide 6 : demo
what I wanna tell : 	Here we can include final results of how those top 4 models performed on the validation dataset of the 15-day labeled data


slide 7 :  blockers
* Approved Cornerstone use case for pulling data 
* Sufficient time for data labeling and preparation 
* Model training time – the larger the dataset, the longer the training time
what I wanna tell: Cornerstone use case - not a blocker for this exercise since we used my use case from my previous team to query the Cornerstone data, but a use case for Digital Experience Design and Engineering  team would be required for us to continue this work on a more serious level
* 	
* 		Sufficient time for training data labeling/preparation, although alternative methods used to mediate this were fairly successful and could still be used
* 		I s


slide 8 : roadmap
* Hyperparameter Tuning 
* Implementing Rule Set on Predictions 
* Focus on Specific Services 
* Production Deployment Logistics 
what I wanna tell : hyper parameter tuning
* avoiding false positives
* 		My thinking for future work in terms of finalizing a more precise determination of an outage is that a rule set should be implemented that takes both 1 and 2 class chat instances into consideration when determining an outage.
* 		ex: detection of a certain threshold of 1 class ‘outage possible’ instances plus at least one 2 class ‘outage likely’ instance within a certain time period
* 		noticed most frequent mention of outages with a select group of services - WDE, GSP, CSP, etc. - look at our manually labeled training dataset to confirm this core group of services
* 		We could fine tune this solution further to focus on these services
* 		Would also like to continue experimentation into XLNet with enough time to resolve prediction blockers from that model
* 		How do we want to deploy and what Ame





make me script for each slide with all the info above used. My time is within 8 minutes
