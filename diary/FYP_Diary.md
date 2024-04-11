# Project Diary 
_(Updated 11/04/24)_

 Much of my notes prior to Week 1 have been handwritten, I hope to transcribe more onto here in due time.


(Prior to this, I was planning for 5 different project titles, will fill this later. Much research was done on ML fundamentals, algorithms and theory here.)

### 18th September 

Allocated supervisor and informed by supervisor of my project title (allocated on 15th September). Supervisor will be Prof. Zhiyuan Luo.
Project will be Comparison of Machine Learning Algorithms.

- Made notes and research on "Breaking the ties" with K-NN algorithms
- Made notes and basic research on Uniformity with Decision Trees, noted measures of uniformity (Gini, Entropy, Info Gain) - will need to read more about these soon.
^^ Both points are highly relevant for the early deliverables.

Watching CS229 ML lecture [video](https://youtu.be/jGwO_UgTS7I?si=TcRudZ_jwvuylEkv) given by Andrew Ng at Stanford, 2018: 
Very easy to absorb lecture on overview of different types of ML. My project so far seems to be directed towards Supervised Learning-Classification



### 19th September

Attended Project Talk given by Dr Argyrios, covered 2 presentations. Useful talk, Argyrios clarified to me personally that LaTeX Project Plan can be in any format, isn't an institution-locked layout.
Heavy emphasis on focusing on project plan.

As of writing, the outline of plan is somewhat structured, but much work needs to be done.

Will meet Prof. Zhiyuan on 25th, guidance given on how to access his office

### 20th September

Started trying to plan my meeting with Prof. Zhiyuan. Many questions to ask, just trying to find the right questions. Noted that there are various paths I can take this project, now considering if I should exclusively focus on Classification.
Noted that it's difficult to research on Classification without needing to read into Regression techniques. There's also lots of notation to digest.

- Understood the functionality of parameters (theta) alongside features/inputs (x). Learning a lot on notation in ML. The algorithms given in the brief seem to be more-or-less non-parametric, however - though I could be mistaken.
- Research on kernel functions and how they transform data into a different representation such that data is easier to evaluate and more separable. I understand this very superficially. Need to read more on examples: Linear, Polynomial, Gaussian (RBF) and Sigmoid Kernels.
- Some basic research on Multi-Class Support Vector Machines("one-vs-all", "one-vs-rest"). Complicated for now, will be useful for final deliverables. Also tribute to Vapnik. 
- Research on RSS (Residual Sum of Squares) Criterion - not sure if relevant to KNN/Trees/Classification algorithms, seems moreso relevant to linear regression
Started watching CS229 [Lecture 2](https://youtu.be/4b4MUYve_U8?si=GvrB1HdXNJ66JWNE), very useful information on notation by Andrew Ng, mostly focused on Regression however. 

- Some concerns that I'm mostly focused on theory and not so much on implementation. Started digging around sci-kit learn [documentation](https://scikit-learn.org/stable/modules/tree.html) today: - will continue to get familiar in upcoming days.

Upon arrival at campus, I've borrowed 5 books from library physically:
1. The Elements of Statistical Learning, 2nd Edition. (TEoSL)
2. Pattern Recognition and Machine Learning by Bishop. 
3. Foundations of Machine Learning by Mohri
4. Introduction to Machine Learning with Python by Muller & Guido. (MLwP)
5. Hands-on Machine Learning with Scikit-Learn and TensorFlow

Primarily I've been using (1) TEoSL as reference and using the others, or the internet to decipher concepts or notation when it's difficult to at first glance.
I've not been using (4) yet but I think it will be vital as I start making proof of concept programs. 

Today found out from Prof. Vovk via Moodle that a new textbook has been released online: An Introduction to Statistical Learning with Applications in Python, 2023. It seems to be a simpler version of TEoSL, so it might be the perfect resource for me. I'm in luck!
Exchanged emails with the library services today but book isn't available physically, they've provided me an online copy. 

GitLab repo is online! Aiming to put this diary on there.

-Research on using OverLeaf (LaTex editor) in sync with GitLab such that I can track version history of my reports. Also research on Git, branches and recapping version control studies. Emailed Prof. Argyrios for confirmation of repo, as well as gitlab classes and slides from earlier talk. 

### 21st September

Made various changes and commits to repo last night and this morning, it now contains folders for this diary and the project plan I'm working on in LaTeX. I've been pushing to master(or main) lately and it doesn't feel right to me - so now that I've made my repo somewhat easy to navigate, I'm making a branch to continue research on: plan

I'll use the planning branch up until my Project Plan is complete, I can then use my repo to explore my research and put proof of concept programs for the algorithms together while I'm still trying to define my project. This branch will represent my initial development phase. I've spent a lot of time last night and this morning keeping up to date with git conventions such that everything is done orderly. Creating this branch felt like the natural thing to do in this formative phase of my project. My intention is to merge the plan branch with main around when the Project Plan is complete and submitted, marking the beginning of development (and likely, a new dev branch).

I purchased a pocketbook for meetings and ideas with my supervisor

I will spend today looking at sci-kit learn and trying to play with library functions that have implements knn and decision trees already. I have barely looked into datasets yet and I must do so, this will likely define the direction of my project. Still need to look into whether Classification is what I want to exclusively work on. I will also work on my Project Plan, this should be a priority.

### 22nd September

Yesterday I continued reading about kernel functions and SVMs and honestly - found myself overwhelmed with theory and how much I still need to understand, let alone bring to the meeting on 25th. I think I need to focus more now on implementation. I'll work on the simplest ML algorithms and focus on implementing them and assessing them, further theory can wait I think.
I aim to now focus on working through the exercises in MLwP and get familiar with working with datasets.
Yesterday I also found Kaggle as a useful resource for datasets. Kaggle is the largest open-source ML community and it's highly likely I'll use their resources in this project, along with UCI and Delve, as long as the data isn't too difficult to process.

Also thinking of turning these diary files into markdown format for easier use.

Setting up python and Jupyter on macOS surprisingly time-consuming.

### 23rd September

Been playing with GUI elements today, tkinter and pyqt5 - I've not really decided how the end-product of this project is going to look and function, so I started investigating this today.
Wasted a ton of time configuring python environments.

Spent good amount of time today planning for meeting, have a decent outline of things to cover. In the process, have structured my approach to tackling the project a little better.

### 24th September

Wrote meeting outline for supervisor before tomorrow's meeting, during the process did a lot of research on model evaluation concepts and metrics

Sent email with a decent amount of detail, very much looking forward to the meeting. I feel this has been a productive week but I need to make many decisions still on how I'm going to tackle this project.

Will likely change these to .MD files tomorrow and perhaps make a research folder. Hoping to start implementing Jupyter Notebook projects next week.

My priority next week however, will be my project plan. 

### 25th September

Met supervisor! Meeting was thoroughly helpful. Main takeaway from meeting was to 'keep it simple'. I will focus on Nearest Neighbours for now and try implementing the simplest variation of this algorithm in handwritten form. Doing this will allow me to fully understand the algorithm at its core.
I'll likely just focus on simple implementations of Nearest Neighbours, Decision Trees and perhaps Logistic Regression - I need to read more into this.

My priority is the Plan right now, I think I know which reports to focus on for this term at least. I should start with mathematical implementations of the algorithms to further my understanding - and this might help while I'm creating the Plan. I should then get started on building the algorithm and can structure my approach in SE terms via the Plan too. It's clearer to me now that the Plan is a tool to help me, rather than just a deliverable for assessment.

### 27th September
Attended FYP Talk on LaTex and Referencing today. Since classes have started, it's gotten a lot busier and less time to focus entirely on research and project progress.

Additions have been made to Project Plan abstract. I hope to have an outline of some kind prepared by Friday and ideally a draft before the end of the week which can be reviewed by my supervisor.

I've forgotten to mention that I've been using a 6th physical book by Tom Mitchell called Machine Learning for much of my research. I find it easier to absorb theory on there, and it covers decision tree theory in much more detail than TEoSL. I think it's probably much more suited for Classification problems.

### 28th September
Following the previous FYP talk, I've started using Google Scholar to find papers to use as references - and already found some great authoritative texts on the NN algorithm.  
I aim to make real headway on the Plan today and I really hope to complete some kind of draft by tomorrow, such that I can get early feedback from my supervisor. Today I really need to make some executive decisions on the direction and goals of my project.  
I've decided it may be wise to include some kind of _success criteria_ in my Abstract, such that I can make critical goals to deliver for the interim review. This project is so open to expansions that it's important to define what should be expected by the end of this. Creating a set of possible expansions may make it more flexible. Once I start assigning a time-frame to this project, it'll become clearer.

### 15th October
It's been a while since the last diary entry.
Unfortunately, I may have underestimated how challenging it would be to balance the project with the other modules I'll be studying.  
The week following my last entry was entirely focused on implementing my Project Plan and submitting something that motivated and outlined my approach to the project sufficiently.  
The week after this should have been focused on developing my proof of concept programs, designing my UMLs and beginning my Nearest Neighbours report. Unfortunately, last week was not fruitful and I'll have to spend this week catching up with my outline immediately.

### 16th October
I've made small progress with the NN algorithm proof of concept and it works as expected. The labs in my machine learning module have been very useful with learning python syntactic sugar and dealing with datasets in Jupyter Notebooks. I'd like to somehow implement some kind of unit testing this week that I can carry forwards for the rest of term.  
The repo needs to progress to a new branch for development and I hope to do this immediately after this entry. I'll be transferring from the 'plan' branch to a 'dev-poc' branch, updating the main branch in the process. It's likely that this will be the first of many dev branches in the overall process.  
I'll need to update my README too with all the new structure I have put in place.   
I'll likely setup some structuring of my interim and supplementary reports(NN) this week.  
I'll move forwards with haste, and hopefully hear feedback on my outline soon, just to find if it's too ambitious, though I feel I'm already seeing that it is. 

### 2nd November
With the start of November, I do feel that I'm falling behind, not only with the work outlined in my plan - but also unfortunately, the diary entries. I do hope to make these more frequent and get back on track with my weekly diary entries.  
  
I've made progress on Nearest Neighbours and K-Nearest Neighbours proof of concept programs in Jupyter Notebooks. I realised quite quickly that it was imperative to create train-test-split functionality immediately just to test these algorithms functionally, and I've managed to do so.  
I think that my implementation of train-test split will help in implementing k-folds cross-validation.  
I've essentially realised that I need to implement model evaluation functions in conjunction with my models otherwise it's difficult to know if I'm going the right direction with my model implementation.  
  
If I focus on completing an implementation Nearest Neighbours and beginning an implementation with the Tree algorithm - which I believe will carry its own challenges - before the end of this week, as well as implementing cross-validation in some way. I may be able to catch up with my plan's outline.  
  
I'm moreso worried about making progress on my reports, as I'm very behind on this and I believe this will be more time-consuming than the algorithm's implementation. I'll be making immediate work on the NN algorithm report, and have already built the skeleton of the report in LaTeX.  
My only consolation with this is that I don't feel lost with the theory of this much at all and I think research should be straightforward thanks to the reading I did out of interest during the summer.  
  
I had my second supervisor meeting a week ago, and the main theme of this meeting was that I essentially need to simply put my head down and deliver. My understanding is there, I simply need to push work forwards without fear of making errors.  
  
Areas that I might need to look into next week include handling missing data in datasets and how to implement PyUnit tests for my validation functions. I've played with a heart disease dataset from the UCI repository and noticed that handling missing values might be a challenge that I need to focus on within my data preprocessing phase.  
  
CS3920 lectures and labs have been handy for me looking ahead, and I think investigating normalisation techniques and how they affect model accuracy may be something to do next term.  
  
All in all, I'm running behind, I'm aware of it, and I need to move quickly to catch up in time for the interim review.
  
### 15th November
I attended the FYP talk today by Prof. Dave Cohen about Presentations and how to bring forward our project during Presentation Week. I'm actually looking forward to talk about the research I've made. However I believe I'll need to make some compromises in order to deliver my targets on time, and I think this will have to take the form of the report deadlines set by myself.   
I've simply not been keeping up with working on reports in adjacent form with my development, along with the intense assignments I'm working on currently this month. I think I'll have to work solely on the interim report and put together my findings on the two algorithms concurrently within my interim report - rather than simply bringing in two completed algorithm reports to introduce within the interim report. Once the interim report submission is complete, I can review if I want to add more detail or background to the algorithm reports during the second term.   
It's likely that I'll need to perform some kind of review before the end of the the year to create a more thorough plan for Term 2.   
  
I've made some progress with the Decision Tree with Gini Impurity but still trying to work out how to introduce stopping criteria. I'm tentative to commit something that breaks.  
From the library, I've picked up the book by Leo Breiman - Classification and Regression Trees. It's verbose but goes into pretty deep detail about tree splitting, stopping and pruning strategies. I'm curious about introducing entropy/information gain but the benefits don't seem obvious to me unless I bring about categorical data into the mix - which I've just not done yet.  

I've discovered a really neat dataset on Kaggle called the Titanic dataset, it seems like a fun thing to bring in and a little more interesting than classifying flowers. I'd like to try and bring this into play before the end of term, but it depends on the difficulty of data preprocessing. I think it'd be nice to talk about for my presentation.  

Looking back at my early research, it's quite funny to see how much of the theory I was reading through is rather irrelevant to the implementation I'm putting together - (RBF Kernels, Multi-class SVMs) - and it makes sense now why my supervisor told me to keep things simple back during our first meeting.
  
### 24th November
Judgement day is almost upon us as my Interim Review deadline is approaching. I believe I have two functional algorithms deemed worthy for evaluation, though both could be extended in various ways.  
I now need to piece together my report and ensure I have sufficient notebooks that evaluate the algorithms' performance. I've not quite completed the cross-validation functionality, but I think I can have this complete in a few days.  
My third supervisor meeting has been scheduled for the 30th.  
I need to ensure I have a testing strategy in place that checks the robustness of the algorithms while I made alterations to them.  
  
I do feel that I could bring in a third algorithm into play during second term, to make things more interesting. I think after learning about SVMs in CS3920, I have an idea on how this could be a third classification algorithm I could use for comparison to KNN and Trees.

### 27th November
Since my last entry I've managed to implement K-Folds Cross-Validation, and made a few tweaks to my models such as adding a get_depth() function to my decision tree. I think my the end of today I can have Leave-One-Out Cross-Validation implemented (as it's essentially when K-Folds = N) with its own get_score functions.   
Implementing K-F CV feels significant as it seems to be the first time that all of these modules are working together, and seeing it work in the notebook has been satisfying, confirming that the data structures are all working as I'd hoped.    
One thing I'm a little disappointed about is how my algorithms don't yet work on datasets with missing data, or with categorical data - which significantly limits which datasets my models can work on. I have to make a decision on whether I can somehow implement this soon, or just focus on delivering my report and interim review.
   
I've managed to implement a more explicit way of handling ties within my KNN algorithm today.
  
I feel that I could find a way to automate some kind of "hyperparameter tuning" for my algorithms - I'm just currently unsure if this would lead to data snooping. From what I'm reading - it's important to use one hold-out set for hyperparameter tuning, and then a different test set for general performance. I'll need to clarify this with my supervisor. 

As for categorical data, I've been reading on how ordinal data and nominal data are generally handled - using one-hot encoding and label encoding. It seems to me that I may need to build some kind of encoder for my data in the future to handle future datasets. I've not considered normalisation - and it'll be necessary for me to implement this, particularly scaling, as if I want to start training on the breast cancer dataset and titanic data - which has missing data, but also a wide range of scales of data - I'll need to bring this forward soon.

On another note, today the deadline for the interim review has been postponed by a week. This is generally good news, but does make me a little confused about whether I should keeping implementing new changes - or wrap things up for review and focus on my report and presentation.

### 28th November
Having read more about normalisation, handling categorical data and missing data - I think it would be wise to delay this until after the review. I could try and quickly hash something together that works, but I do feel that I need to thoroughly research and plan this implementation out in such a way that they're integrated with model models smoothly.  
  
For KNN, there's a chance that I may need to implement a new distance measure such as Hamming Distance or Jaccard Distance to handle categorical data sufficiently, without making misleading effects to the distance between samples. 

### 29th November
I've managed to add various test suites to thoroughly check edge cases for the functionality implemented thus far. Doing this helped me notice gaps in my exception handling when passing values around - such as in K-folds and train-test-split when specifying the size of the fold or split. 

### 30th November
Today I finally had my third meeting with my supervisor. It went smoothly and my supervisor seems satisfied with my progress thus far. I managed to clarify doubts I had about the following:  
  
- Handling missing and categorical data:
  - For missing feature data, it's usually find to remove the entire sample for that instance. As long as there's enough data samples to make training sufficient.  
  - I asked about implementing an encoder to handle nominal and ordinal data, this seemed like a suitable idea.  
  - I asked about how this may affect the distance function in KNN, and if nominal data could be handled correctly, if I should using Hamming, Jaccard or Cosine function as an alternative. He suggested I keep trying with Euclidean as this should work okay with the situations I'm dealing with. He reminded me to just try it before I doubt the implementation.

- He cleared up misunderstandings I had about other performance metrics besides accuracy - such as the difference between Precision and Recall, and using the F1 score. I think I could quickly calculate these with the implementation I have so far. He also reminded me that I can find the variance of my model by analysing the range of accuracy values I find during K-Folds Cross-Validation.  
- Hyperparameter Tuning - I asked about the procedure of finding the ideal hyperparameters of KNN and Trees, i.e. the number of neighbours K for KNN and the maximum depth of the tree constructed. He reminded me to avoid data snooping, which I seemed to be doing in Notebook 6. I need to keep the test set separated and utilise a validation set for this.  
- He emphasised that it's very important that there's some form of data normalisation for KNN, as it's rather distance-sensitive. This may have explained unusually low values I was getting for the optimum k-value on large datasets such as ionosphere (though perhaps this is the curse of dimensionality in effect).  
  
Following the meeting, I immediately implemented a MinMaxScaler so that I have some form of data normalisation I can use to handle misaligned data ranges, particularly for KNN. I will need to correct my hyperparameter tuning procedure too.  
Overall in project development, I have built a pretty strong foundation for me to use for Term 2. I just need to show the results found on the three datasets I've chosen - iris, ionosphere and banknote authentication.  
Right now, my priority is putting the presentation and report together - this needs more urgent work. 

### 6th December
Obligatory diary entry - I've been working away at completing my deliverables for the interim review - the interim report and the presentation. The presentation has been submitted and will take place on the 8th.  
I'm finding it difficult to not add further functionality to the project last minute. I think there's still a lot I can do to automate various processes, such as the hyperparameter tuning process, and finding metrics such as accuracy. I think I can work on the preprocessing.py module this month after the review, and also work on integrating the process of calculating accuracy, recall and precision within a new metrics.py module.  
My report is coming together, it leans quite heavily into the theory. Overall, I do feel quite proud of what I've managed to do this term, and I think I can carry this momentum forwards, once submission is completed.
  
### 10th December
Term 1 is over! I gave my interim presentation on the 8th and it went very well, I received around four questions - two from the audience and two from the Chair. One member of the audience suggested I implement integer division into the way I calculate my folds for K-Folds Cross-Validation. I've realised now that I have already implemented this, but the need to handle the remainder is still necessary. I received further questions about how my MinMaxScaler was implemented, and why I chose the algorithms that I focused on.  
  
Lately, I've been thinking about what I may need to research on for the rest of this month to make next term a little easier for me.

### 17th December
I think the most immediate addition I'd need to make to my project should be the addition of some form of Encoder to handle missing data and categorical data. Handling this would quite immediately increase my algorithm's functionality and I'd quite rapidly be able to double the number of datasets I work with, and evaluate my algorithms a lot more rigorously.  
  
The Leave-One-Out Cross-Validation exposed just how inefficient my tree algorithm is during training - and I think I'd need to find ways to make the algorithm less greedy during the split search. I could also do some research on the ID3.5 algorithm to see how it differs to the CART algorithm that I've implemented.  
Another significant issue I need to bring my attention to, is how the tree model doesn't reveal much information about its structure, such as the number of training points that were classified at each leaf node. Finding this will make debugging and optimisation significantly easier, and could potentially help me devise a method of finding a confidence level for each prediction.  
  
A key objective in the next term will be to work on an interface for these models to work, and essentially, act as the front-end of my final project. I've not made a start on this, so it'll be wise for me to at least research which libraries I'll use to put this together.  
  
Pipelines were mentioned towards the end of the CS3920 ML course - I feel like this could be a solid addition to my implementation. The interim submission shows each process executed manually - and this simply won't be feasible in a graphical interface layout in Pipelines. I think pipelines could help me to wrap up all the preprocessing in a neat way, as well as potentially show off some intricate software design practices.  

I've yet to use Precision and Recall as metrics in my project, and it seems silly to only use accuracy as a sole metric so far. However I've noted that these metrics are generally only used in binary classification - so it'll be necessary to adapt this for multi-class classification. 
  
### 16th January
I'm gearing up for the upcoming term and I've comprised a list of necessary objectives:
  1) Preprocessing - Handling missing and categorical data to accommodate more datasets
  2) Tree Refinement - Implement optimisations to how the tree searches for splits during the training process. Alter the tree model structure so that the leaf nodes are more informative.
  3) Metrics - Introduce more model evaluation metrics that allow more thorough comparison and insight into the algorithms' performance. 
  4) Pipelines - Implement pipelines that can encapsulate much of the data manipulation into few executions. This will make it easier to implement an interface for the end-user. 
  5) GUI - Make a head start in implementing a simple but effective graphical interface for end-user to use the algorithms being discussed. Ideally allow for inputted datasets to be analysed and trained upon. 
  6) Implement a third algorithm given there's enough time to do so. Strongly considering logistic regression.  
    
I'm relieved to have received positive feedback on my project's progress so far, as well as the presentation - It's assuring to know that my implementation so far is going the right direction.  
  
I'm likely to have a lot more clarity over next steps after my next supervisor meeting. 
  
### 25th January
Just had my 4th meeting with Zhiyuan - I found it quite helpful. Every meeting has been focused around the importance of delivering a finished product - and it's reminded me that I need to define the scope of what I want to achieve this term with everything working smoothly. Zhiyuan recommended I try and implement a third algorithm quickly and then focus on implementing the final product.  
I've place a lot of emphasis on components like data preprocessing but my supervisor recommended that I don't waste too much time on this, it can be easy to waste time on this and it can become convoluted.  
I need to implement a third algorithm quite quickly, I'm leaning towards Logistic Regression for now - a parametric model. I think it'd also be interesting to introduce Grid Search to my implementation in addition to what's been previously considered. 
My supervisor reminded me to focus on discussing my results and findings of my project - and evaluate their performance. It's important that my report reflects its title, as obvious as that sounds.
  
I'll pushing to the repo again very soon. 

### 27th January
Worked on a new simple notebook just to experiment with SKLearn's preprocessing libraries and envision how I could make a preprocessing pipeline.  
  
I've been looking for material to research on Logistic Regression so I can get working on it quickly. I've found a great excerpt from chapter 4 of 
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition. I think this could be helpful.  

### 9th February
Somewhat hit a bit of a roadblock in research this week: Implementing a Softmax Regression Classifier.  
It's clear to me that I need to implement a form of multi-class logistic regression, and the ideal form of this seems to be utilising softmax as the decision function. However, I've been getting pretty lost with how to implement this and all the inner mechanisms of it.  
I've somewhat concluded that I'll start with the basics and work my way up - implementing binary classification with a Logistic Regression Classifier on simple data, perhaps working with One-Vs-Rest to create a form of multi-class classification, and then from there - hopefully implementing softmax will be more intuitive.  
  
I want to finish algorithm implementation asap so that I can focus on metrics and GUI, and approach a final working implementation. So far, I've managed to implement Notebook 8 with a basic implementation of LR on Iris data. As well as a new file for my LR implementations logistic_regression.py.

### 16th February
The Softmax Regression Classifier has been implemented! I now have three working algorithms that can be evaluated and I feel jubilant about it. Breaking down the algorithm bit by bit definitely helped.  
  
I began by developing a binomial classification implementation, with basic logistic regression implemented - and this made it far simpler to visualise how to put together a multi-class softmax regression classifier. Gradient descent is explicitly implemented with a loss history to track the loss with every iteration of training. 


## 22nd February
I've been working on the wine quality dataset in a jupyter notebook of the same name. It's a pretty challenging dataset to work with and allows me to put my three models to the test and against each other.
  
## 6th March
I've been working on preprocessing a lot since my last entry. I've been significant delayed by other deadlines unfortunately, I need to make a headway in wrapping up this project into a final product, with an interface. 

## 8th March
I've been working on testing my preprocessing tools on a dataset, so I've been working on a notebook under the name of preprocessing. I've implemented a preprocessing pipeline, as well as a one-hot and ordinal encoder and an imputter for handling missing data. The imputation has been challenging to work with, especially with many data types. I've been using a dataset from the UCI repository called 'Congressional Voting Records' which is almost entirely categorical, with missing data and no numerical data at all - so it's a good example to test out my preprocessing tools.

I've noted that I might need to implement a label encoder for my models, I don't think my models have sufficient handling for when my labels are not just integers. I'm also thinking of automating the confusion matrix display function, it'll save much hassle when putting these jupyter notebooks together.

## 13th March
I've made significant progress with my preprocessing module and have made a CombinedPreprocessor that can combine preprocessors for different types of data (numerical, ordinal and categorical). This works a bit similar to SKLearn's ColumnTransformer functionality, and allows transformation of specified feature columns in the dataset. Combining these transformations into one object abstraction makes it easier to pass between cross-validation functions.  
I've finally been working on the interface and showing this functionality on the desktop. So far, the implementation is very simple and much more needs to be put together to allow all kinds of datasets to be loaded and trained upon, but I've managed to show something that isn't just a Jupyter Notebook. I hope to collect some more datasets and clean up the dataset folder a little, allow preset choices and find a way to automate the preprocessor - or simply allow the end-user to select how the dataset should be preprocessed on separate columns. It's been a fruitful week.  

## 14th March
Today I've managed to make the UI far more robust with threading using QThread from PyQT5's library. The interface can take a numerical dataset, and uses a numerical preprocessor to handle missing points and scale the data. The models are then trained and evaluated with 5-Folds Cross-Validation and the results are displayed with confusion matrices below. It's been time consuming getting this to work robustly and passing signals between the thread and the window.  
I still need to work on implementing the CombinedPreprocessor so that I can throw more datasets at this, and then working on getting this to work on a validating set that can actually tune the models. The models are chosen rather ad-hoc for now. I'd really like to somehow quickly implement grid-search in the poc side so that I can tune an ideal instance of each model before they're evaluated.  
I want this interface to show more metrics from the datasets the models have been trained on, for further performance analysis. 

## 19th March
The last couple of days have been very productive with the UI implementation, with various functionalities having been added and working together.  
  
 Since the last entry, I've managed to implement multiple threads working asynchronously during the tuning process. The preprocessing and tuning functionalities are working with their corresponding buttons. I've reached a point where my application now displays the grid search of the models during hyperparameter tuning.
   
There are still issues here and there with exception handling of data types and the progress bar doesn't work consistently. I've managed to implement the backend for the metrics calculations - I now need to add this to the UI.
  
## 21st March
My supervisor meeting is today! I'll be working on displaying my metrics onto the interface before then and clearing up the issues made in the previous diary entry.  
I'll synchronise the tuning functionality with the progress bar and the abort button. I also need to find the overall accuracy from my k-folds cross validation scores and append that to the metrics.  
I hope to demo my app to my supervisor and take any necessary feedback with me such that I can adapt as needed.

## 22nd March
My supervisor meeting went relatively well, and my supervisor had no qualms overall over the functionality and depth of my app. Rather, he emphasised the importance of my report and how it must effectively back up and demonstrate every aspect of the application. I've lots of work to do on the report. The way that results are shown seem fine, but he reminded me to make a user manual to explain the interface to him, as this was not trivial to him.
  
This is the only meeting I failed to record but we did manage to talk about the report a little more and he reminded me to implement a suitable professional issues section as well.

## 24th March
Assessing what's left to implement, I very much want to add the ability for the user to change parameters in both the evaluation and tuning process. It shouldn't be too complicated for the former.
  
I'd very much like a logo for my project while it runs, rather than the blank white page I see on MacOS. My little sister has offered to help with this, as a design student herself, and is making mockups for the app now.
  
I'll be spending the next week or two, focusing more on the report than the implementation. Other modules require a lot of my attention right now so I expect a break in commits until a lot of this has been completed. My priority is shifting towards updating the interim report with new material and current progress. 
  
My code for the UI implementation is a mess and needs refactoring. I think I can not only refactor this but implement a design pattern called MVC. This pattern seems to fit perfectly with my application, but it would involve making some new modules to stow away the thread classes away from the UI elements. I'd probably have to change the way that some of the UI app works to keep ui.py only focused on UI elements. I generally need to tidy much in my repo.
  
Finally, I need to make decisions about which datasets to use in my findings and results. I'll try to select datasets that reflect a representative interpretation of the models' performance. 

## 4th April

I've been very bogged down by other projects and assignments but I've kept a focus on writing my report and wrapping up the app alongside. I've been giving some though of how to handle the app's distribution. I've considered building an .exe file but I'm having serious issues with running it in a stable way, with path issues. I'm having worse issues with making it work on Windows. I've also considered using PyPi and allowing the user to install it with "pip install". And finally, I suppose it's not a terrible backup plan just to let the user run the source code, but I'd rather not resort to this.  
  
The logo is ready! And I hope to use this in accordance with the apps as they are distributed.
  
I hope to refactor the repository very soon and add missing research materials before submission.

## 11th April
It's been a hectic day! It's the end of the road. I may add one more diary entry after submission for the sake of my own personal reflection. All in all, just trying to make sure I submit a stable finished product in time.
  
I've learned a tremendous amount from this project. Perhaps the most important lesson has been - simply just deliver. Don't worry about breaking anything, just experiment and see what sticks.
  
I do hope this will be the first of many future machine learning projects for myself personally, and I can imagine continuing to work on this on beyond submission.
