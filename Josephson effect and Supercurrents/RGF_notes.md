## Useful trouble-shooting tips:

Recently read the first few sections of this paper https://arxiv.org/pdf/cond-mat/0501609.pdf, which gives a slightly more detailed exposition of the recursive green's function method explained by Ando.

A few useful things to try:
* A better expression for the group velocity of the nth propagating mode is: 
$ v_{n} = -\frac{2a}{\hbar}\textrm{Im}\big[ \lambda_{n}(\pm) \vec{u}^{\dagger}_{n}(\pm) \hat{B}^{\dagger} \vec{u}_{n}(\pm)\big] $ 
where $\hat{B}^{\dagger}$ is the hopping matrix to the left. The group velocity of evanescent modes is automatically zero.
* The $\hat{F}(\pm) = \hat{U}(\pm) \hat{\Lambda} (\pm) \hat{U}^{-1}(\pm)$ transfer matrix should also satsify:
$ (\varepsilon - \hat{H}) -\hat{B}^{\dagger}\hat{F}(\pm) -\hat{B}\hat{F}^{-1}(\pm) = 0$
It would be good to see if this is the case.
* In the present paper, the authors define dual eigenvectors $\tilde{u}_{n}(\pm)$ such that $ \tilde{u}^{\dagger}_{n}(\pm) \vec{u}_{m}(\pm) = \delta_{n,m} $ and $  \tilde{u}_{n}(\pm) \vec{u}^{\dagger}_{m}(\pm) = \delta_{n,m} $. I think this is equivalent to taking the inverse of the $\hat{U}(\pm)$ matrix...
* They also normalise the $\vec{u}_{n}(\pm)$ vectors!
* Finally, clearly the transmission matrix you are getting is clearly unphysical, because you are including the effects of evanescent states. The physical transmission matrix must be obtained by multiplying by:
$$\sqrt{\frac{v_{R , n}(+) }{ v_{L, m}(+)}}$$
where $n , m$ index eigenmodes. This also eliminates all matrix elements $t_{nm}$ between unphysical evanescent modes. 
* Lastly, functions like `scipy.linalg.eig` solves the generalised eigenvalue problem- perhaps better accuracy?