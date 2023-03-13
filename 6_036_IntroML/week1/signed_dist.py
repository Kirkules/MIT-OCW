# Author: Kirk Boyer
# Week 1 Homework for MIT's Openlearning Introduction to Machine Learning
# https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week1/week1_homework/

import numpy as np

length = np.linalg.norm  # course website provides length, so let's not rewrite it

def signed_dist(x, th, th0):
    return (np.dot(th.T, x) + th0)/length(th)
