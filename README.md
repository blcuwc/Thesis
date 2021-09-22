This is Chen Wang's thesis project.

First supervisor: Suzan Verberne  

Second supervisor: Anne Dirkson

This project started at Mar, 2020 and ended at July, 2021.

Title: Using ensemble models with structural information in social media to aid rumour stance classification

Abstract: In previous research, using structural information to address stance classification in conversational social media has been limited to a single sequential model. In this work, we explore the benefit of constructing ensemble models with sequential and non-sequential models to aid rumour stance classification. This is a multi-classification task that aims to predict user's attitudes towards a rumour post. During the research process, we successfully adapt the original BranchLSTM method to make it able to receive various sizes of tweet conversations and make it available to the public. We experiment with combining the sequential model BranchLSTM and the pre-trained model DistilBERT by merging their predicting probabilities to the same tweet and using one-step and two-step voting classifiers to classify the new probability features. Through single-model experiment analysis, we find out that DistilBERT has advantages in addressing a more balanced dataset comparing to BranchLSTM, but BranchLSTM could perform better in predicting the large categories of the imbalanced dataset. Through ensemble experiment analysis, we prove that the ensemble model could learn both features of BranchLSTM and DistilBERT with the help of a voting classifier, but is not able to keep advantages from both deep learning models. Finally, through comparison between sequential voting classifier and non-sequential classifier, we find out that sequential voting classifier could make use of the context but may have less strength in classifying a single simple tweet comparing to the non-sequential classifier.

