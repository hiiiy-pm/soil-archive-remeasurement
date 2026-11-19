# Integer allocation and comparator scope

`code/integer_allocation.py` independently implements the bounded separable objective

$$
\min_{n_h}\sum_h A_h/n_h,\qquad
A_h=N_h^2\sum_p a_p s_{hp}^2,
$$

with integer counts, stratum capacities, fixed total budget and lower bounds that retain previously paid samples. Adding one record reduces the objective by `A_h / [n_h (n_h + 1)]`. The allocator repeatedly chooses the largest feasible reduction. Exact priority ties follow stratum order.

The mathematical allocation objective is credited to the multi-parameter framework of Yang, Shepherd, Lumley and Shaw, *Optimal Two-Phase Sampling Designs for Generalized Raking Estimators With Multiple Parameters of Interest*, Statistics in Medicine (2026), https://doi.org/10.1002/sim.70786. This implementation is original project code, not copied author R code. No third-party software source or third-party software licence is incorporated by this module. Publication citation does not imply endorsement by those authors.

The count lower bound is `max(previously_selected, min(stratum_size, 2))`. Singleton strata can be enumerated completely; a budget equal to the sum of feasible lower bounds is accepted. A budget below the floor, or an unavailable variance proxy, returns an explicit fallback flag. The experiment's caller then fills counts using its specified metadata-only balanced rule. Such budgets are retained and logged. No free observations are added to make the floor feasible.

The soil-archive comparators adapt this count allocator to influence contributions or LOO residual contributions for predefined treatment, time and interaction contrasts. They do not reproduce the source article's complete logistic/Cox influence functions, generalized raking estimator, or adaptive inclusion-probability calculations. The final retained-reference reconstruction is a separate estimator; the allocation objective does not guarantee its MSE or sign recovery.

The active-sampling comparator is separately credited to the cited Imberg et al. framework. It uses square-root model-based second moments and a pilot LOO prediction gate, adapted here to unique-record sampling without replacement and the common reconstruction. Guarantees for another sampling scheme or estimator are not transferred to this adapter.
