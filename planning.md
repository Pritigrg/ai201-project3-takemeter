# TakeMeter Planning Document

## 1. Project Overview

TakeMeter is a text classification project that evaluates the type and quality of online football discourse. For this project, I will build a classifier using public posts/comments from **r/soccer**, a large Reddit community focused on football discussion. Each text example will be classified into one of four labels: **informational**, **analysis**, **opinion**, or **noise**.

---

## 2. Community

I chose **r/soccer** because it is an active, text-heavy football community with frequent discussion about matches, players, managers, tactics, transfers, referees, results, and fan reactions. This community is a good fit for a classification task because the discourse varies widely: some posts share factual updates, some comments explain football decisions in detail, some users express strong opinions, and others post jokes, insults, or low-effort reactions.

This variation makes the classification problem interesting because the task is not simply separating “good” and “bad” comments. A fan opinion can still be relevant to the community even when it is not deep analysis, while a factual update may be useful even if it does not express an opinion. The classifier needs to learn the difference between information, reasoning, opinion, and low-value noise.

---

## 3. Label Taxonomy

I will use four mutually exclusive labels.

### Label 1: informational

**Definition:** A post or comment is labeled **informational** if it mainly shares a factual update, official news, match result, question, or discussion prompt without making a strong argument or emotional judgment.

**Example 1:**  
“Team USA becomes the second team to qualify for the knockout stages after defeating Australia 2-0.”

**Example 2:**  
“What’s your ideal final?”

### Label 2: analysis

**Definition:** A post or comment is labeled **analysis** if it explains a match event, rule, referee decision, team performance, player role, tactic, or qualification scenario using reasoning, evidence, or specific football details.

**Example 1:**  
“Tiebreaker is head-to-head this World Cup, which means USA, Mexico, and Turkey have nothing to play for in the last game.”

**Example 2:**  
“Australia should have had a penalty because the defender pushed the attacker and the ball hit his arm in an unnatural position.”

### Label 3: opinion

**Definition:** A post or comment is labeled **opinion** if it expresses a personal reaction, prediction, praise, criticism, complaint, or hot take but does not provide detailed reasoning or evidence.

**Example 1:**  
“This Turkey performance reminded me of 2022 and 2018 Spain but worse.”

**Example 2:**  
“I really like this 48-team format because smaller countries get to enjoy the World Cup.”

### Label 4: noise

**Definition:** A post or comment is labeled **noise** if it is very short, vague, sarcastic, meme-like, insulting, off-topic, or does not add much meaningful discussion value.

**Example 1:**  
“LMAO what a joke.”

**Example 2:**  
“Goodnight”

---

## 4. Hard Edge Cases

One difficult edge case is the boundary between **analysis** and **opinion**. For example, “Chelsea looked terrible because their midfield was lost” gives a reason, but the explanation is still general and does not include specific football details. I will label this as **opinion** unless the comment gives clearer reasoning, such as mentioning a formation, tactical adjustment, player role, rule, statistic, substitution, or specific match event.

Another difficult edge case is the boundary between **informational** and **opinion**. A title such as “Spain 0-0 Cabo Verde is the first goalless draw of the World Cup” is informational because it mainly states a fact. However, a post such as “Spain 0-0 Cabo Verde proves Spain are overrated” would be labeled **opinion** because it makes a judgment.

A third difficult edge case is the boundary between **opinion** and **noise**. For example, “This team is finished” is a reaction, but it is also short and unsupported. I will label it as **noise** if the comment is mostly a one-line joke, insult, vague reaction, or meme. I will label it as **opinion** if it expresses a clear football viewpoint that could reasonably start discussion.

During annotation, I will use the following rule: if the text mainly states or asks something, label it **informational**; if it explains something using football reasoning, label it **analysis**; if it expresses a clear judgment or reaction, label it **opinion**; if it is too vague, sarcastic, insulting, or low-content to add discussion value, label it **noise**.

---

## 5. Data Collection Plan

I will collect at least **200 public posts/comments** from r/soccer. I will focus on discussion-heavy sources, including match threads, post-match threads, transfer discussions, referee decision threads, player performance discussions, and daily discussion threads. I will avoid collecting usernames or personal information and will store only the text, label, source/thread type, dataset split, and whether AI pre-labeling was used.

My target label distribution is approximately balanced:

| Label | Target Count |
|---|---:|
| informational | 45–55 |
| analysis | 45–55 |
| opinion | 45–55 |
| noise | 45–55 |

If one label is underrepresented after collecting 200 examples, I will collect additional comments from thread types where that label is more common. If **analysis** is underrepresented, I will collect more from post-match threads, referee/rule discussions, or detailed player-performance discussions. If **noise** is underrepresented, I will collect more from live match threads where short reactions and jokes are common. If **informational** is underrepresented, I will collect more post titles, official updates, or discussion prompts. If **opinion** is underrepresented, I will collect more fan reaction and prediction comments.

I plan to split the final dataset as follows:

| Split | Number of Examples |
|---|---:|
| Train | 140 |
| Validation | 30 |
| Test | 30 |

The training set will be used to fine-tune the model, the validation set will be used to check model behavior during development, and the test set will be held out for final evaluation.

---

## 6. Evaluation Metrics

I will evaluate the classifier using **accuracy**, **precision**, **recall**, **F1-score**, and a **confusion matrix**.

Accuracy is useful because it shows the overall percentage of examples classified correctly. However, accuracy alone is not enough because some labels may be easier to predict than others, and the dataset may not be perfectly balanced. For this task, per-class metrics are important because I need to know whether the model performs well across all four categories, not only the most common one.

I will use **precision** to measure how often the model is correct when it predicts a label. This matters because if the model predicts **analysis**, I want that prediction to actually represent meaningful football reasoning. I will use **recall** to measure how many true examples of each label the model successfully finds. This matters because a useful classifier should not miss most analytical comments or misclassify most noise. I will report **macro F1-score** because it balances precision and recall while treating all labels equally.

I will also include a confusion matrix to show which labels the model confuses most often, such as whether it mistakes **opinion** for **analysis** or **noise** for **opinion**.

---

## 7. Definition of Success

For this project, I would consider the classifier successful if it achieves at least **70% overall accuracy** on the test set and a **macro F1-score of at least 0.65**. Macro F1 is important because it gives equal weight to all four labels, even if one label appears more often than another.

For a real community tool, I would want stronger performance before deployment. A genuinely useful deployed version should achieve at least **80% accuracy**, a **macro F1-score of at least 0.75**, and no individual label should have an F1-score below **0.65**. This matters because the tool should not only work well overall but should also avoid consistently failing on one category, especially **analysis** or **noise**.

At the end of the project, I can objectively determine whether I met my success criteria by comparing the test-set accuracy, macro F1-score, per-class F1-scores, and confusion matrix against these thresholds.

---

## 8. Baseline Comparison Plan

I will compare my fine-tuned model against a zero-shot baseline using Groq’s **llama-3.3-70b-versatile** model. The zero-shot baseline will receive the label definitions and each test example, then it will predict one of the four labels without being trained on my dataset.

Both models will be evaluated on the same test set. This comparison will help show whether fine-tuning a smaller model on my specific labels performs better than using a large language model with only prompting.

---

## 9. AI Tool Plan

### Label Stress-Testing

Before annotating the full dataset, I will use an AI tool to stress-test my label definitions. I will give the AI my four labels, definitions, and edge-case rules, then ask it to generate 5–10 borderline r/soccer-style comments that are difficult to classify. If I cannot classify those examples cleanly, I will revise my label definitions before labeling all 200 examples.

The goal of this step is not to use AI-generated comments in my dataset. The goal is to find weaknesses in my taxonomy early.

### Annotation Assistance

I may use an LLM to pre-label a small batch of collected examples before reviewing them myself. If I do this, I will treat the LLM labels only as suggestions, not final labels. I will manually review every label before including the example in the dataset.

To track this process, I will include an extra column in my dataset called `ai_prelabeled`. This column will be marked `yes` if an AI tool suggested the initial label and `no` if I labeled the example fully manually. I will disclose this in the AI usage section of my README.

### Failure Analysis

After evaluating the model, I will give an AI tool the list of wrong predictions and ask it to identify patterns in the errors. For example, I will look for whether the model confuses short analytical comments with **opinion**, or whether sarcastic comments are often mislabeled as **noise** or **opinion**.

I will verify any AI-suggested failure patterns myself by checking the original text, true labels, predicted labels, and confusion matrix. I will only include a failure pattern in my evaluation report if I can confirm it with specific examples from my test set.

---

## 10. Final Notes

The main challenge in this project is not only training a model but also defining labels that are clear, meaningful, and consistently applicable. I will focus on making the label boundaries specific enough that I can annotate comments consistently and evaluate the classifier honestly.
