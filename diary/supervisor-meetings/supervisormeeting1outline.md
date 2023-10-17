# Supervisor Meeting 1 Outline

Formatting is less awkward in email, keeping MD file for reference

[Meeting Notes here](https://www.notion.so/Supervisor-Meeting-1-eb487a577da14ceb9617dc8d2d733c6e?pvs=21)

**Outline:**

1. Introduction
• Looking forward to getting to know you and working with you as my supervisor!
• Small intro about myself, I'm returning to studies after a long time, there are challenges this might bring this year
2. Defining the project

I'll be working on '*[Comparison of Machine Learning Algorithms](https://projects.cs.rhul.ac.uk/List2023.php?PROJECT-TYPE=Full)*'. I've informally divided my approach as follows:

Theory
• Algorithms: I've been thinking of focusing on Classification algorithms, as have been outlined in the specification - Nearest Neighbours and Decision Trees, and then hopefully expanding to implement kernel methods and including SVMs - but specifically for classification
    

- I imagine that it's difficult/not possible to compare algorithms designed for regression with those for classification, but you may correct me if I'm wrong about this. For simplicity's sake, I think I'll focus on classification and the evaluation metrics and methodologies that come with that.
    ◦ I'd like some direction if this is not ambitious enough, or if I'm complicating things unnecessarily  
    ◦ Any further suggestions for algorithms to be explored are very welcome

-Comparing Algorithms: I've been reading about concepts such as bias-variance tradeoff, model complexity and overfitting/underfitting. In the context of DTs, I've read about regularisation (and pruning). I've also been trying to read about methodologies such as using a hold-out test set and cross-validation (and it's variations). And also been trying to read about metrics in the context of classification

 - Accuracy, Precision + Recall, F1 Score - but I still have lots to read into to truly understand. I think the wisest thing I should do is to start doing exercises in Jupyter Notebooks using sckit-learn to implement these methodologies and understand them better. I'm aiming to do this asap.

Generally, I've been diving into a lot of theory lately - and it's been a little overwhelming. I think I've realised recently that it might be best to keep the algorithms simple and focus on implementation and solid evaluation.

The books I've been using are mentioned in my repository, the main ones I've been using a lot are:

- The Elements of Statistical Learning. Hastie et al. 2009 - Very exhaustive but also quite intense. Been trying to read in other places and then coming back to this.

- Machine Learning by Tom Mitchell. 1997 

- This feels a little more accessible to me, has decent chapter on decision trees- Introduction ot Machine Learning with Python by Muller and Guido, 2017. 

- Easiest to follow with! But I'm yet to spend time with this, however I feel this may be the most useful book for me with regards to implementation

- There's one more book that I've yet to start reading through but I think it may be vital this year - found [here](https://www.statlearning.com/) - An Introduction to Statistical Learning: with Applications in Python. Only published a few months ago but it's essentially a simpler version of the first book mentioned here in the list, with examples in Python! I discovered it via Prof. Vovk's list of resources on Moodle.

Implementation
• I think I feel the least confident about this aspect of the project, any advice on this is appreciated! I'm quickly trying to learn the best approach for this
• Initially had intentions of using Python, as it seems to be the best supported language in Data Science. My overall idea is to create a python program that functions locally and allows the user to load a dataset, and outputs various insights on the model's performance on the data.
    ◦ The spec suggests using Java or C++ for this - would it be better to explore these languages instead of Python for building the algorithms? I feel quite proficient in Java, but unsure where to go with this.
    ◦ I'm very unsure of how to approach the GUI aspect of this. Currently I'm considering using PyQT5(/6) or TKinter to incorporate a GUI. This isn't a big priority for me this early on, but it's an aspect I don't have a lot of confidence on at this point.
    ◦ Datasets: Alongside UCI and Delve that have been suggested in the spec, I've found that [Kaggle](https://www.kaggle.com/datasets), [OpenML](https://www.openml.org/search?type=data&sort=runs&status=active) and [huggingface](https://huggingface.co/datasets) to have many contributed datasets to experiment on. I think
• Implementing Modern SE Principles/Implementation Life Cycle (Final Deliverables:5) - I'm a little hesitant on how to incorporate this
    ◦ Do you have any advice on how to incorporate TDD in this kind of project? What would be an example of a unit test here?
    ◦ I imagine using UML diagrams may be useful in showing how datasets will be manipulated when inputted?

Deliverables/Assessment
• My primary focus is on the Project Plan due on October 6th
    ◦ Do you have any advice/opinions on using LaTeX? I plan to use [Overleaf](https://www.overleaf.com/) for my Plan and hope to use it for future reports. There's an empty skeleton template on Gitlab. 
    ◦ I need to work on developing an outline, which requires me to make decisions on my implementation.
    ◦ Any suggestions on risks + mitigations that I may not have thought of?
    ◦ Do let me know of your availability (mentioned below) so I know how to plan my timeline a bit
    ◦ Advice on the making the abstract is welcome! Machine learning is such a widely talked about topic now, I'm not sure how much of an introduction to it I should make.
• Reports: Following the early deliverables, I hope to make two reports:1) Nearest-Neighbour Algorithms (1NN + KNN) with strategies to break the ties. I've been reading about breaking the ties with relation to K-Neighbours (weighted voting, distance-weighted, etc)2) Decision Tree Algorithms described with multiple measures of uniformity. So, I've been reading about uniformity and *trying* to read about its measures (Gini Impurity, Entropy, Information Gain).Upon doing further research, I've realised that it may be wise to do another report on model performance and evaluation metrics. This project is about comparing machine learning algorithms - so it makes sense to me to explore the theory behind how to do that - specifically for simpler classification algorithms.3) Evaluation metrics and concepts for learning algorithms tasked towards Classification - I need to define this better but essentially a report on the material discussed above in Theory: Comparing Algorithms. I think there's a lot here that could be explored in a report regarding how to properly evaluate model perfomances and writing a report (or dividing this area into multiple reports) may help me when putting my interim and final report together.
• Proof of Concept: I think it makes sense to start making basic proof of concept programs asap:
    ◦ Perhaps work through scikit-learn exercises in textbook on notebook to get familiar with models and methodologies - and keep in repo for reference
    ◦ 1NN algorithm with basic/artificial dataset
    ◦ KNN algorithm with basic/artificial dataset
    ◦ Basic Tree algorithm with basic/artificial dataset
    ◦ Expand all prior three with larger and larger datasets
    ◦ A GUI proof of concept program (even with dummy functionality) may be useful for me to figure how to approach the UI element.
• As time passes, I can continue focusing on these, expand on these and use the knowledge I learn from building these to build my final product. If I do introduce more algorithms, I can incorporate them in a similar manner.

Your availability
• I'm not too familiar with the usual supervisor-student protocol, but it'd be helpful to me to know how is the best way to ask you for assistance or to bring up concerns. Please do let me know if and I'm asking more than I should be. It'd be helpful to know how often I can arrange meetings with you, even beyond the 5 mandatory meetings in the year
• Do you have a preference of face-to-face or online meetings?
• Would you be willing to give me some feedback on my Project Plan next week? Or perhaps the week after?