JumpCut
=======

Source code of SIGGRAPH Asia '2015 paper *"JumpCut: Non-Successive Mask Transfer and Interpolation for Video Cutout"* by Qingnan Fan *et al.*

Project page: http://irc.cs.sdu.edu.cn/JumpCut/

Note
----

Due to the complexity of our interactive real-time system, the interaction part is not included in our distribution. And the interpolation is also part of our system, it's not within these codes either.

These codes include the main contribution of our algorithm, splitNNF, edge classifier and levelset. We provide a sample code for you to test. You can modify it based on this to fit your need.

Note that the edge map is computed with the codes [1] provides. You need to calculate the edge map with their codes first or you can also merge their MATLAB codes with ours using MATLAB Engine. In OpenCV 3.0.0, Dollar edge map is included, using their implementation is also a good choice.

[1] DOLL´AR, P., AND ZITNICK, C. L. 2013. Structured forests for fast edge detection. In Proc. ICCV, IEEE, 1841–1848.

Dependencies
------------

Our system is running under Cuda 6.5 and OpenCV 2.4.10 in visual studio 2013. If you want to change the configuration of our solution file, you can modify jumpcut.props and reload the solution file in visual studio.

Cite
----

You can use our codes for research purpose only. And please cite our paper when you use our codes.

```
@Article{Fan:2015,
Title = {JumpCut: Non-Successive Mask Transfer and Interpolation for Video Cutout},
Author = {Qingnan Fan and Fan Zhong and Dani Lischinski and Daniel Cohen-Or and Baoquan Chen},
Journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH ASIA 2015)},
Year = {2015},
Volume = {34}
Number = {6},
}
```

Contact
-------

If you find any bugs or have any ideas of optimizing these codes, please contact me via fqnchina [at] gmail [dot] com