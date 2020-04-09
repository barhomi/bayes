# bayes

In order to practice bayesian modeling, I am using this book/tutorial:
https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers

The code in this repository is my reimplementation of the tutorial. I am not adding anything new. Re-implementing something
helps me understand it better. 


To run the code (make sure you are in the directory where the Dockerfile lives and change SRC_DIR to where the bayes folder live):
```shell script
 SRC_DIR=$HOME/src/bayes
 docker build --tag tf_proba .
 docker run -it -p 8888:8888 -v $SRC_DIR:/tf/bayes tf_proba
```