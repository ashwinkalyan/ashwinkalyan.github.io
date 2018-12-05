---
layout: page
permalink: /diff-beam-search/
---
# $$\nabla$$BS: Differentiable Decoding for Neural Sequence Models
## Submitted by: Ashwin Kalyan [as part of CS7643: Deep Learning] <br> <br>

## What problem did you solve? <br> <br>
Neural sequence models like RNNs, LSTMs and Transformers are widely used to model sequential tasks like captioning, translation, planning, etc. 
Given a trained model, decoding the $$K$$ most likely sequences i.e. the inference problem is often solved using greedy heuristics like Beam Search (BS). 
As observed by previous works, outputs decoded via BS are uninteresting and very similar; rendering the expressivity of the deep network largely under utilized.
Further, there is a _disconnect_ between the training and inference procedures as: <br> 
1) (_train-test_ mismatch) the model is not exposed to its own predictions during training and <br> 
2) (_loss-evaluation_ mismatch) evaluation criteria is often not correlated to the objective being optimized for. <br>
In this work, we propose $$\nabla$$BS, a _trainable_ version of beam search that addresses the above mismatches by closely integrating the training and inference phases.
## How is it done today? What are the limitations of current practice? <br> <br>
While beam search  still remains the most widely used decoding algorithm, trainable decoding algorithms have been introduced by [Gu et al.](https://www.aclweb.org/anthology/D17-1210) and [Chen et al.](https://arxiv.org/pdf/1804.07915.pdf)
Importantly, these methods are limited to decoding the _best_ sequence and cannot be extended trivially to handle top-K decoding. 
Further [Goyal et al.](https://arxiv.org/pdf/1708.00111.pdf) propose a continuous relaxation of beam search that enables end-to-end training; however, it still suffers from train-test mismatch as test-time decoding is still non-differentiable.
More recently, [Negrinho et al.](https://papers.nips.cc/paper/8264-learning-beam-search-policies-via-imitation-learning.pdf) formulate a beam search policy by learning scoring functions that estimate the value of extending a beam. 
This method poses the additional overhead of designing a suitable scoring function class and moreover, is incapable of modeling repulsive interactions between beams.
## Who cares? If you are successful, what difference will it make? <br> <br> 
As discussed above, beam search generally produces uninteresting output lists and so, developing an end-to-end approach for top-K inference will address this defeciency. 
Further, as found by [Kalyan et al.](https://arxiv.org/abs/1610.02424) diversity is crucial for many applications (like image captioning) and BS often fails to capture the variability present in the output space. 
Another scenario where BS fails is program synthesis where a strict grammar dictates the sequence; [Devlin et al.](https://arxiv.org/abs/1703.07469) for instance find that BS is sub-optimal in finding the top-K programs that satisfy certain specifications.
The hope is that a _learnt_ inference procedure performs better than BS in controling for diversity and syntax in the scenarios discussed above. 

## Is anything new in your approach? <br> <br>
__Key Intuition.__ The novelty in our approach is mainly got by looking at beam search as a sequential subset selection problem. 
At each time step, the K partial solutions maintained by the algorithm can be extended using any token $$y\in\mathcal{V}$$ where $$\mathcal{V}$$ is the vocabulary.
In other words, there are $$K\times|\mathcal{V}|$$ possible extensions and beam search selects the top-K by sorting the partial solutions according to their joint log-likelihood under the trained model. 
Instead of sorting according to the joint log-likelihood, we propose to _learn_ a function that does the $${K\times|\mathcal{V}|\choose K}$$ problem at each time $$t\in\{1,2,\dots T\}$$.
## What did you do exactly? How did you solve the problem? <br> <br>
__How to learn to select subsets?__ Subset selection is often used in extractive summarization where we need to choose K sentences out of a total of N sentences. 
Further, we need to choose sentences that are _representative_ of the entire passage and at the same time convey relevant information. 
In fact, we intend to solve a very similar problem in the top-$$K$$ decoding setting at each time step -- out of the $$K\times|\mathcal{V}|$$ extensions, we want to select K elements that lead to good quality sequences and yet, different from each other.
Not surprisingly, we adopt the formalism in text summarization literature and therefore, use cardinality constrained submodular function maximization to perform subset selection. 

__Parametrized Differentiable Subset Selection__. A set function $$f:2^\mathcal{V}\rightarrow\mathbb{R}_{\geq 0}$$ is submodular if for sets $$\mathcal{S}\subseteq\mathcal{T}\subseteq\mathcal{V}$$ and $$e\in\mathcal{V}\backslash\mathcal{T}$$, $$f(S\cup\{e\}) - f(\mathcal{S}) \geq f(\mathcal{T\cup\{e\}}) - f(\mathcal{T})$$. 
A submodular function is (a) _monotone_ if $$f(\mathcal{T})\geq f(\mathcal{S})$$ and (b) _normalized_ if $$f(\emptyset) = 0$$. 
We are interested in finding the maximizer, $$\mathcal{S}^* = \arg\max_{\mathcal{S}\subseteq\mathcal{V}, |\mathcal{S}|\leq K} f(\mathcal{S})$$. 
If the set function $$f(\cdot)$$ is parametrized by $$\beta$$, [Tschiatschek et al.](https://www.ijcai.org/proceedings/2018/0379.pdf) provide a way to learn the parameters via gradient descent.
Their algorithm works iteratively by sampling from the distribution $$SOFTMAX\left(\{\Delta(e|A_{k-1})\}_{e\in\mathcal{V}\backslash A_{k-1}}\right)$$ for $$k\in\{1,2,\dots K\}$$ where $$\Delta(e|\mathcal{S}) = f(\mathcal{S}\cup\{e\}) - f(\mathcal{S})$$.
This idea coupled with Deep Submodular Functions, a parametrized class of submodular functions proposed by [Bilmes et al.](https://arxiv.org/pdf/1701.08939.pdf) provide a way of _learning a parametrized submodular function via gradient descent based algorithms._

__Training via Imitation and Reinforcement.__ Ideally, we would like to use REINFORCE proposed by [Williams et al.](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) to backpropagate through the non-differentiable metric of interest. 
However, since the action space at each time step is exponential i.e. it is the power set of all possible extensions, it is important to start from a reasonably good initialization. 
For this purpose, we train via imitation learning by using standard beam search outputs for _supervision_.
This ensures that the initialization gives a sequence policy that is at least as good as the _expert_ policy which in our case is beam search. 
As mentioned, the model is then trained via REINFORCE to address loss-evaluation mismatch.

## Why did you think it would be successful? <br> <br>
Tightly integrating training and inference addresses a fundamental issue in the currect procedure of training neural sequence models. 
As the proposed method does this in a reasonably well-motivated manner and uses suitable techniques from reinforcement learning to bridge loss-evaluation mismatch, the approach is promising at an intuitive level.

## How did you measure success? <br> <br> 
Given an evaluation metric $$\phi(\hat{y}, \{r_1, r_2, \dots r_N\})$$ that compares a prediction $$\hat{y}$$ against a set of ground truths $$\{r_i\}_{i=1}^N$$, we draw from the facility location problem to evaluate a list of predictions $$\{\hat{y}\}_{k=1}^K$$.
Therefore, we care about for $$\sum_{r}\max_{\hat{y}}\phi(\hat{y}, r)$$.
From now on, we refer to this measure as fac-score i.e. if the the given metric is CIDER, we use fac-CIDER and so on.
In practice, we are interested in achieving high values in this measure on an unseen test set and further expect to outperform standard BS.

## What experiments were used? What were the results, both quantitative and qualitative? <br> <br>
The preliminary results were obtained for the image captioning task on the Flickr-8k dataset that contains 8000 images and 5 captions each. 
The dataset was divided into 6000/1000/1000 for train/val/test respectively. 
We start with a trained RNN that treats the image as the first token in the sequence [Vinyals et al.](https://arxiv.org/abs/1411.4555). 
The image features are extracted using a 152-layer ResNet traied on imagenet.
For the inference policy, we use a two layered DSF and expand to only the top 100 extensions at each step.
We train using a beam size of 20 and use, BS outputs as supervision in the imitation learning phase and as baseline reward in the reinforcement learning stage.
All models were trained using SGD with a learning rate of 4e-4 and momentum of 0.9. 
Early stopping was used to select the best model by looking at the training and validation performance.
The models were implemented in [pytorch](https://pytorch.org/).

We find that the our method achieves a fac-CIDEr value of 2.56 outperforming BS (2.12) and imitation-only baseline (2.25). 
A careful analysis of the performance w.r.t. image complexity similar to [Kalyan et al.](https://arxiv.org/pdf/1610.02424.pdf) is yet to be carried out.
Further, for the sake of significance, we need to demonstrate this performance gain on other datasets like Flickr-30k and COCO.
