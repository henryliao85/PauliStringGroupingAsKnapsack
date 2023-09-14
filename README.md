# PauliStringGroupingAsKnapsack
A trial to map grouping problem of Pauli strings to knapsack problem, and use known genetic algorithm (NSGA-II) to solve it.

# Problem setup and mapping
Given $n$ Pauli strings, and the goal is to identify the smallest number of groups among these Pauli strings.

The constraint for a group is that every members in the same group has to commute with each other.

Then we can consider a gene of length $n$ that labels which group each string belongs to.

For example, when $n=3$, a gene can be a list of integers of size $3$ such as [1,1,3]. This represents the first and second strings belong to group 1 while the third one belongs to group 3.

Then by adopting NSGA-II to the design of gene, we can find the smallest grouping given the runtime is long enough.
