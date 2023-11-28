# Project Diary 
_(Updated 15/11/23)_

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