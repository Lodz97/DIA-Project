# DIA-Project 2020
Project for the Data Intelligence Application course at Politecnico di Milano

### Pricing + Advertising

 
The goal is modeling a scenario in which a seller exploits advertising tools to attract more and more users to its website,
thus increasing the number of possible buyers. The seller needs to learn simultaneously the conversion rate and the 
number of users the advertising tools can attract.
#### Goals
1. Imagine:
    * one product to sell;
    * three classes of users, where, for every user, we can observe the values of two binary features 
    (feel free to choose the features and their domains);
    * the conversion rate curve of each class of users;
    * three subcampaigns, each with a different ad, to advertise the product, and each targeting a different class 
    of users;
    * there are three abrupt phases;
    * for every abrupt phase and for every subcampaign, the probability distribution over the daily number of clicks for 
    every value of budget allocated to that subcampaign.

2. Design a combinatorial bandit algorithm to optimize the budget allocation over the three subcampaigns to maximize 
the total number of clicks when, for simplicity, there is only one phase. Plot the cumulative regret.

3. Design a sliding-window combinatorial bandit algorithm for the case, instead, in which there are the three phases 
aforementioned. Plot the cumulative regret and compare it with the cumulative regret that a non-sliding-window 
algorithm would obtain.

4. Design a learning algorithm for pricing when the users that will buy the product are those that have clicked 
on the ads. Assume that the allocation of the budget over the three subcampaigns is fixed and there is only one phase 
(make this assumption also in the next steps). Plot the cumulative regret.

5. Design and run a context generation algorithm for the pricing when the budget allocated to each single subcampaign 
is fixed. At the end of every week, use the collected data to generate contexts and then use these contexts for the 
following week. Plot the cumulative regret as time increases. In the next steps, do not use the generated contexts, 
but use all the data together.

6. Design an optimization algorithm combining the allocation of budget and the pricing when the seller a priori knows #]
that every subcampaign is associated with a different context and charges a different price for every context. 
Suggestion: the value per click to use in the knapsack-like problem depends on the pricing, 
that depends on the number of users of a specific class interested in buying the product. 
Notice that the two problems, namely, pricing and advertising, can be decomposed since each subcampaign 
targets a single class of users, thus allowing the computation of the value per click of a campaign only
on the basis of the number of clicks generated by that subcampaign. Plot the cumulative regret 
when the algorithm learns both the conversion rate curves and the performance of the advertising subcampaigns.

7. Do the same of Step 6 under the constraint that the seller charges a unique price to all the classes of users.
Suggestion: for every possible price, fix this price and repeat the algorithm used in Step 6.
Plot the cumulative regret when the algorithm learns both the conversion rate curves and the performance of
the advertising subcampaigns.


## Basket4All
The chosen product is the subscription to a streaming platform of sport.
In the details it offers different championships of basket played in distinct countries.

We differ the users in three classes splitting on two feature: the Country where they live and their Gender.


|               | Class1 | Class2 | Class3 |
| ------------- |------  | -----  |------  |
| COUNTRY       | USA    | Europe | Any    | 
| GENDER        | Male   | Male   | Female |

We distinguish three different phases:
* Phase 1: the launch of the product
* Phase 2: the enter in the market of a new competitor
* Phase 3: the updating of the product with new features to offer