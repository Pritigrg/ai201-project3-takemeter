# TakeMeter: r/soccer Discourse Classifier

## Project Overview

TakeMeter is a text classification project that evaluates the type and quality of online discourse in an online community. For this project, I built a classifier for posts and comments from **r/soccer**, a Reddit community focused on football news, match discussion, tournament reactions, player debates, referee decisions, and fan opinions.

The goal of the classifier is to assign each post or comment to one of four labels:

- `informational`
- `analysis`
- `opinion`
- `noise`

The project includes a manually labeled dataset, a zero-shot baseline using Groq's `llama-3.3-70b-versatile`, and a fine-tuned `distilbert-base-uncased` model.

---

## Community

I chose **r/soccer** because it is an active, text-heavy football community with a wide range of discourse quality. Some posts share factual match updates, some explain rules or tactics, some give emotional fan opinions, and others are jokes, sarcasm, or low-effort reactions.

This makes r/soccer a good fit for a classification task because the discourse is varied enough to create meaningful label boundaries. The classifier needs to learn the difference between factual information, reasoned football analysis, personal opinion, and comments that add little value.

---

## Label Taxonomy

I used four mutually exclusive labels.

### `informational`

A post or comment is labeled **informational** if it mainly shares a factual update, match result, official news, question, or discussion prompt without making a strong argument or emotional judgment.

**Examples:**

- “Team USA becomes the second team to qualify for the Knockout stages after defeating Australia 2-0.”
- “What’s your ideal final?”

### `analysis`

A post or comment is labeled **analysis** if it explains a match event, rule, referee decision, player performance, team performance, tactic, or qualification scenario using reasoning, evidence, or specific football details.

**Examples:**

- “Tiebreaker is H2H this World Cup, which means USA, Mexico and Turkey have nothing to play for last game sadly.”
- “Australia should have had a penalty because the defender pushed the attacker and the ball hit his arm in an unnatural position.”

### `opinion`

A post or comment is labeled **opinion** if it expresses a personal reaction, prediction, praise, criticism, complaint, preference, or hot take, but does not provide detailed evidence or technical reasoning.

**Examples:**

- “This Turkey performance reminded me of 2022 and 2018 Spain but worse.”
- “I really like this 48-team format because countries like Curaçao and Haiti being able to enjoy this event for the first time is worth the quality of individual games dropping a bit.”

### `noise`

A post or comment is labeled **noise** if it is very short, vague, sarcastic, meme-like, insulting, off-topic, or does not add much meaningful discussion value.

**Examples:**

- “Ref is blind.”
- “Hydration breaks or as I like to call them: mini retirements.”

---

## Dataset

I collected and labeled posts/comments from **r/soccer**. The dataset was saved as a single CSV file and was not pre-split before loading into the notebook. I did not store usernames or personal identifying information.

The cleaned dataset contains **221 labeled examples**.

### Label Distribution

| Label | Count | Percentage |
|---|---:|---:|
| opinion | 105 | 47.5% |
| analysis | 50 | 22.6% |
| informational | 33 | 14.9% |
| noise | 33 | 14.9% |
| **Total** | **221** | **100.0%** |

No label accounts for more than 70% of the dataset, so the dataset does not violate the assignment's imbalance rule. However, the dataset is still somewhat opinion-heavy, which affected the fine-tuned model's behavior.

---

## Hard Edge Cases

Real r/soccer discourse often mixes facts, questions, opinions, jokes, and football reasoning. I used the main purpose of each post/comment to decide the label.

### Edge Case 1: Question vs. Analysis

**Example:**  
“Wait, I don't get it. Why are Haiti and Turkey already considered officially eliminated? Whether they can actually do it or not, isn't it mathematically possible for them to still finish 3rd in their respective groups, or am I missing something here?”

**Possible labels:** `informational` or `analysis`

**Decision:** I labeled this as `informational` because the main purpose is asking a question and starting a discussion. It mentions a qualification scenario, but it does not provide the explanation itself.

### Edge Case 2: Analysis Mixed with Opinion

**Example:**  
“It's because FIFA have changed the rules for tiebreakers. Head-to-head results are now considered before goal difference. Just another thing FIFA have screwed up, in my opinion.”

**Possible labels:** `analysis` or `opinion`

**Decision:** I labeled this as `analysis` because the main part explains the rule change and why the qualification outcome happened. The final sentence is opinionated, but the comment still provides useful rule-based reasoning.

### Edge Case 3: Opinion With a Reason

**Example:**  
“I really like this 48 teams format because I think that countries like Curaçao, Haiti etc being able to enjoy this event for the first time is massively worth the quality of individual games dropping a bit.”

**Possible labels:** `opinion` or `analysis`

**Decision:** I labeled this as `opinion` because it gives a personal preference about the tournament format. It includes a reason, but it does not analyze a specific rule, match event, tactic, or performance in detail.

### Edge Case 4: Sarcasm vs. Opinion

**Example:**  
“Like primary school football, the keeper is allowed to handle the ball anywhere in their own half.”

**Possible labels:** `opinion` or `noise`

**Decision:** I labeled this as `noise` because the comment is mainly sarcastic and vague. It does not clearly explain a rule, match event, or argument in a way that adds much discussion value.

---

## Model

I fine-tuned **`distilbert-base-uncased`** for four-class text classification on **Google Colab** (T4 GPU runtime), using the HuggingFace `transformers` `Trainer` API. I chose DistilBERT because it is faster and lighter than full BERT while still being strong enough for text classification.

### Training Approach

The dataset was loaded from a CSV file, cleaned, and converted from string labels into integer label IDs using this label map:

| Label | ID |
|---|---:|
| informational | 0 |
| analysis | 1 |
| opinion | 2 |
| noise | 3 |

The data was split 70% / 15% / 15% (stratified by label, `random_state=42`) into **155 train / 33 validation / 34 test** examples. Text was tokenized with a max length of 256 and dynamic padding.

### Hyperparameter Decisions

I trained for **3 epochs** at a **learning rate of 2e-5** with a **batch size of 16** (plus `weight_decay=0.01` and `warmup_steps=50`), selecting the best checkpoint by validation accuracy.

My main decision was to keep **epochs low (3)**. My initial reasoning was that with only 155 training examples, more epochs would risk overfitting and memorizing the training set. In practice the opposite happened: the model badly *under*-fit, never predicting the `informational` or `analysis` classes and producing near-random ~0.26 confidence on almost every example (see [Failure Analysis](#failure-analysis)). This tells me that for this small, imbalanced dataset, 3 epochs at lr 2e-5 gave the freshly-initialized classification head too little signal to separate the four classes — so if I retrained, I would increase epochs and/or the learning rate and add class weighting, rather than keeping the setup "conservative."

---

## Zero-Shot Baseline

For the baseline, I used Groq's **`llama-3.3-70b-versatile`** in a zero-shot setting. The model received the community description, label definitions, and one example per label, then classified each test example without being trained on the dataset.

The prompt required the model to output only one of the four valid label strings:

- `informational`
- `analysis`
- `opinion`
- `noise`

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|---|---:|
| Zero-shot Groq `llama-3.3-70b-versatile` | 0.7647 |
| Fine-tuned `distilbert-base-uncased` | 0.5588 |

The zero-shot baseline outperformed the fine-tuned DistilBERT model by about 20.6 percentage points. This was an important result because it showed that fine-tuning a smaller model on a small dataset did not automatically outperform prompting a larger model with clear label definitions.

---

## Baseline Results

### Baseline Per-Class Metrics

| Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| informational | 1.00 | 0.60 | 0.75 | 5 |
| analysis | 1.00 | 0.25 | 0.40 | 8 |
| opinion | 0.67 | 1.00 | 0.80 | 16 |
| noise | 1.00 | 1.00 | 1.00 | 5 |
| **Accuracy** |  |  | **0.76** | 34 |
| **Macro Avg** | 0.92 | 0.71 | 0.74 | 34 |
| **Weighted Avg** | 0.84 | 0.76 | 0.73 | 34 |

### Baseline Reflection

The baseline performed well overall, especially on `opinion` and `noise`. However, it struggled with `analysis`, which had very low recall. This means that when the baseline predicted `analysis`, it was usually correct, but it missed many true analysis examples.

My hypothesis is that the zero-shot model was conservative about using the `analysis` label. It may have classified short rule explanations or lightly reasoned football comments as `opinion` because many r/soccer comments mix reasoning with fan-style language.

---

## Fine-Tuned Model Results

### Fine-Tuned Per-Class Metrics

| Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| informational | 0.00 | 0.00 | 0.00 | 5 |
| analysis | 0.00 | 0.00 | 0.00 | 8 |
| opinion | 0.58 | 0.94 | 0.71 | 16 |
| noise | 0.50 | 0.80 | 0.62 | 5 |
| **Accuracy** |  |  | **0.56** | 34 |
| **Macro Avg** | 0.27 | 0.43 | 0.33 | 34 |
| **Weighted Avg** | 0.35 | 0.56 | 0.43 | 34 |

### Fine-Tuned Confusion Matrix

Rows are true labels. Columns are predicted labels.

| True Label \ Predicted Label | informational | analysis | opinion | noise |
|---|---:|---:|---:|---:|
| informational | 0 | 0 | 4 | 1 |
| analysis | 0 | 0 | 6 | 2 |
| opinion | 0 | 0 | 15 | 1 |
| noise | 0 | 0 | 1 | 4 |

The confusion matrix shows that the fine-tuned model never predicted `informational` or `analysis`. Most errors came from true `informational` and `analysis` examples being predicted as `opinion`.

---

## Failure Analysis

The fine-tuned model made **15 wrong predictions out of 34 test examples**. The main failure pattern is that the model collapsed toward the two labels it learned most easily: `opinion` and `noise`. It predicted many examples as `opinion`, even when the true label was `analysis` or `informational`.

The clearest confusion pattern was:

| True Label | Common Wrong Prediction | Pattern |
|---|---|---|
| `analysis` | `opinion` | Comments with reasoning were treated as personal takes. |
| `informational` | `opinion` | Questions or factual observations were treated as opinions. |
| `analysis` | `noise` | Longer analytical comments with casual language were sometimes treated as low-effort. |
| `opinion` | `noise` | Emotional fan comments with slang or emojis were sometimes treated as low-effort. |

The biggest issue was the `analysis` → `opinion` confusion. Many soccer comments contain both football reasoning and personal judgment, so the model often focused on the opinion-like wording instead of the reasoning structure. For example, comments about Uruguay lacking creativity, Australia possibly deserving a penalty, or Oyarzabal needing space all include football-specific reasoning, but the model predicted `opinion`.

The second issue was that the model did not learn the `informational` label well. Questions such as “Is Japan a top 10 national team in the world currently?” or “Is Doku expected to miss the rest of the tournament?” were misclassified as `noise` or `opinion`. This suggests the model did not strongly learn that questions and discussion prompts should be classified as `informational`.

The confidence scores for the wrong predictions were low, mostly around **0.26–0.27**. This suggests the model was uncertain rather than confidently wrong. That is useful because it shows the model had not learned a clear boundary between the labels.

This appears to be both a data distribution problem and a label-boundary problem. The dataset contains more `opinion` examples than the other labels, and the boundary between `analysis` and `opinion` is genuinely difficult. To improve the model, I would collect more short `analysis` examples, more `informational` question examples, and more hard boundary examples where a comment contains both personal judgment and football reasoning.

---

## Misclassified Examples

| Post | True Label | Predicted Label | Confidence | Analysis |
|---|---|---|---:|---|
| “With the lack of creativity Uruguay has in the final third I wouldn't be surprised to see cape Verde getting a draw” | `analysis` | `opinion` | 0.27 | This failed because the comment includes a prediction, which makes it look like an opinion. However, the prediction is based on a football-specific reason: Uruguay's lack of creativity in the final third. The model focused more on the prediction language than the reasoning. |
| “Is Japan a top 10 national team in the world currently?” | `informational` | `noise` | 0.26 | This failed because the post is short, so the model likely treated it as low-effort. However, it is actually a discussion question asking for evaluation of Japan's current status, so it fits the `informational` label. More short informational questions in training could help. |
| “So I watched the replay and I don’t understand why Australia didn’t get a penalty when Irankunda went up for the header, then the defender pushed him and the ball just falls straight onto the American...” | `analysis` | `opinion` | 0.26 | This failed because referee complaints often sound opinionated in soccer discourse. However, this example gives specific evidence from the play: the header, the push, the arm position, and the lack of appeal. The correct label is `analysis` because the comment reasons through a referee decision. |
| “It’s kappa logo, quite shit” | `noise` | `opinion` | 0.27 | This failed because the comment does express a negative opinion, but it is extremely short and does not add meaningful discussion value. Under my taxonomy, short vague insults or low-effort reactions should be `noise`, not `opinion`. |
| “Oyarzabal needs to be in space like that making runs. He’s a lively player not someone that can live in the box.” | `analysis` | `opinion` | 0.26 | This failed because the statement has a player evaluation, which may look like opinion. However, it explains the player's role and movement, so it should be classified as `analysis`. The model missed the football-specific reasoning. |

---

## What Would Fix These Errors?

To reduce these errors, I would make four changes:

1. Add more short `informational` examples, especially questions and discussion prompts.
2. Add more short `analysis` examples where the reasoning is only one or two sentences.
3. Add more hard boundary examples between `analysis` and `opinion`.
4. Try class weighting or oversampling so the model does not over-predict the larger `opinion` class.

The most important fix would be improving the `analysis` label. The fine-tuned model needs more examples showing that a comment can include a prediction or criticism but still be `analysis` if the main purpose is football reasoning.

## Sample Classifications

| Post | True Label | Predicted Label | Confidence |
|---|---|---|---:|
| “That hurts.” | `noise` | `noise` | 0.27 |
| “yamine lamal, yamal lamine, yalamine yamal, do you see what i'm saying?” | `noise` | `noise` | 0.27 |
| “Ronaldo is a legend, was a great player a few years back but now it's just sad how he's holding Portugal back. It was already the case in 2022, but it's worse now.” | `opinion` | `opinion` | 0.27 |
| “With the lack of creativity Uruguay has in the final third I wouldn't be surprised to see Cape Verde getting a draw.” | `analysis` | `opinion` | 0.27 |
| “You think he works as a sub because he is not being started ever😭. That’s like Mbappe only ever being subbed in and you saying well he works as a joker sub.” | `opinion` | `noise` | 0.27 |

The correct `noise` prediction for “That hurts.” is reasonable because the post is extremely short and does not provide factual information, football reasoning, or a clear discussion-worthy opinion. The correct `opinion` prediction for the Ronaldo example is also reasonable because the post expresses a personal judgment about Ronaldo’s current role with Portugal rather than giving detailed tactical evidence.

The two incorrect examples show the main weakness of the fine-tuned model. The Uruguay/Cape Verde example was labeled as `analysis` because it connects a prediction to a football reason: Uruguay’s lack of creativity in the final third. However, the model predicted `opinion`, likely because the phrase “I wouldn't be surprised” sounds like a personal take. The substitute-role example was labeled as `opinion`, but the model predicted `noise`, likely because the emoji and casual wording made it look more low-effort than it actually was.

---
## What the Model Captured vs. What I Intended

I intended the model to learn four discourse categories: factual information, football analysis, personal opinion, and low-effort noise. However, the fine-tuned model mostly learned a simpler distinction between `opinion`-like comments and `noise`-like comments.

The model captured some obvious low-effort comments, but it missed the deeper semantic difference between `analysis` and `opinion`. It also failed to recognize `informational` posts. This suggests that the model overfit to surface-level signals such as emotional wording, sarcasm, and short reaction style rather than learning the full label taxonomy.

This result was useful because it showed the gap between what I intended the model to learn and what it actually learned. It also showed that the label definitions were understandable to a large zero-shot model, but the fine-tuned DistilBERT model needed more balanced and targeted training data to learn the same distinctions.

---

## Definition of Success

For this class project, I considered the classifier successful if it achieved at least **70% overall accuracy** and a **macro F1-score of at least 0.65** on the test set.

The fine-tuned model did **not** meet this success criterion. It achieved 55.88% accuracy and a macro F1-score of 0.33. The baseline model did meet the accuracy threshold and performed better overall.

For a real community tool, I would want at least **80% accuracy**, a **macro F1-score of at least 0.75**, and no individual label with an F1-score below **0.65** before deployment.

---

## Spec Reflection

One way the spec helped guide my implementation was by requiring a clear label taxonomy before training. This forced me to define exactly what counted as `informational`, `analysis`, `opinion`, and `noise`, instead of using a vague good/bad classification.

One way my implementation diverged from the ideal plan is that the fine-tuned model did not outperform the zero-shot baseline. The original goal was to create a useful fine-tuned classifier, but the results showed that the model struggled with minority labels and collapsed toward `opinion` and `noise`. This divergence was useful because it revealed a real limitation in the dataset and training setup rather than hiding a failed result.

---

## AI Usage

I used AI assistance in several parts of this project.

First, I used AI to help refine my label taxonomy. I initially considered longer labels such as “Match / Rule Analysis” and “Fan Reaction / Hot Take.” After reviewing the dataset, I simplified the final labels to `informational`, `analysis`, `opinion`, and `noise`. I accepted this change because the shorter labels were easier to use consistently and easier for both the model and evaluation code to parse.

Second, I used AI to help identify patterns in the model's wrong predictions. The AI pointed out that the fine-tuned model never predicted `informational` or `analysis`, and that many true `analysis` examples were being predicted as `opinion`. I verified this myself by checking the confusion matrix and per-class metrics.

Third, I used AI to help draft parts of the README and planning document. I edited and reviewed the generated text to make sure it matched my actual dataset, labels, results, and interpretation.

**Annotation disclosure:** I labeled the dataset manually but also used an LLM to assist during annotation. For some examples, I asked an LLM to suggest a label based on my definitions; I treated those suggestions only as a starting point and personally reviewed and corrected every label before including it in the dataset. The final label on every example reflects my own judgment, not the model's — in the difficult `analysis` vs. `opinion` cases especially, I overrode AI suggestions that leaned toward `opinion` when the comment actually contained football-specific reasoning.

---

## Limitations and Future Work

The biggest limitation of this project is the small and somewhat imbalanced dataset. Although no label exceeded 70% of the dataset, `opinion` was still the largest label. This likely encouraged the fine-tuned model to over-predict `opinion`.

Another limitation is that the distinction between `analysis` and `opinion` is genuinely difficult. Many football comments contain both evidence and emotion, so the model needs more examples that clearly demonstrate this boundary.

In future work, I would:

1. Collect more `analysis` and `informational` examples.
2. Add more hard boundary examples between `analysis` and `opinion`.
3. Try class weighting or oversampling for minority labels.
4. Increase the test set size.
5. Compare DistilBERT with a stronger model such as RoBERTa.

---

## Demo Video

▶️ **[Watch the demo (3–5 min)](https://www.loom.com/share/d00800008e394c878c666f84ef93abd0)**
<!-- 
The walkthrough covers:

- The fine-tuned DistilBERT model classifying sample r/soccer posts, with the predicted **label** and **confidence** shown for each.
- **One correct prediction** narrated — why the model got it right (e.g. the Ronaldo `opinion` example, which is personal judgment with no tactical evidence).
- **One incorrect prediction** narrated — why it went wrong (e.g. the Uruguay/Cape Verde example, where `analysis` collapsed into `opinion` because "I wouldn't be surprised" reads as a personal take).
- A brief walkthrough of the **evaluation report**: 55.9% fine-tuned accuracy vs. 76.5% zero-shot baseline, the confusion matrix showing `informational` and `analysis` never being predicted, and the ~0.26 confidence pattern that points to under-fitting. -->