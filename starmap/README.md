# StaRMAP.jl

Julia implementation of the StaRMAP solver originally proposed by B. Seibold and M. Frank[^1].


# Installation

Generate a personal access token and on the RWTH GitLab instance. Save it, as the website won't reveal it again!
run
```
pkg> add https://git.rwth-aachen.de/epma_project/starmap.jl.git
```
and follow the prompts, using your GitLab username (not your TIM-ID!) and the access token you generated.

For SSH access, try setting the environment variable `JULIA_PKG_USE_CLI_GIT=true`, then:
```
pkg> add git@git.rwth-aachen.de:epma_project/starmap.jl.git
```



# References
[^1]: B. Seibold and M. Frank, “StaRMAP - A second order staggered grid method for spherical harmonics moment equations of radiative transfer,” arXiv:1211.2205 [math-ph, physics:physics], Jun. 2014, Accessed: Dec. 05, 2021. [Online]. Available: http://arxiv.org/abs/1211.2205
