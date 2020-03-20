torch.Size([15480, 100])
| Epoch: 1 | Train Loss: 1.026 | Train Acc: 48.89 | Val. Loss: 0.923 | Val. Acc: 59.05 | Test Loss: 0.911 | Test Acc: 60.50 |
| Epoch: 2 | Train Loss: 0.955 | Train Acc: 56.63 | Val. Loss: 0.906 | Val. Acc: 61.91 | Test Loss: 0.882 | Test Acc: 63.57 |
| Epoch: 3 | Train Loss: 0.928 | Train Acc: 58.29 | Val. Loss: 0.917 | Val. Acc: 58.97 | Test Loss: 0.890 | Test Acc: 60.75 |
| Epoch: 4 | Train Loss: 0.905 | Train Acc: 59.95 | Val. Loss: 0.893 | Val. Acc: 61.13 | Test Loss: 0.857 | Test Acc: 61.70 |
| Epoch: 5 | Train Loss: 0.896 | Train Acc: 59.90 | Val. Loss: 0.863 | Val. Acc: 61.92 | Test Loss: 0.822 | Test Acc: 65.90 |
| Epoch: 6 | Train Loss: 0.877 | Train Acc: 61.58 | Val. Loss: 0.863 | Val. Acc: 63.04 | Test Loss: 0.819 | Test Acc: 64.28 |
| Epoch: 7 | Train Loss: 0.870 | Train Acc: 61.66 | Val. Loss: 0.846 | Val. Acc: 64.60 | Test Loss: 0.808 | Test Acc: 64.73 |
| Epoch: 8 | Train Loss: 0.850 | Train Acc: 62.72 | Val. Loss: 0.830 | Val. Acc: 63.92 | Test Loss: 0.788 | Test Acc: 66.27 |
| Epoch: 9 | Train Loss: 0.848 | Train Acc: 62.83 | Val. Loss: 0.832 | Val. Acc: 65.30 | Test Loss: 0.794 | Test Acc: 66.60 |
| Epoch: 10 | Train Loss: 0.838 | Train Acc: 63.37 | Val. Loss: 0.826 | Val. Acc: 65.30 | Test Loss: 0.782 | Test Acc: 67.26 |
| Epoch: 11 | Train Loss: 0.828 | Train Acc: 63.67 | Val. Loss: 0.840 | Val. Acc: 64.60 | Test Loss: 0.804 | Test Acc: 65.84 |
| Epoch: 12 | Train Loss: 0.814 | Train Acc: 64.59 | Val. Loss: 0.812 | Val. Acc: 65.91 | Test Loss: 0.776 | Test Acc: 67.31 |
| Epoch: 13 | Train Loss: 0.799 | Train Acc: 65.43 | Val. Loss: 0.803 | Val. Acc: 65.56 | Test Loss: 0.769 | Test Acc: 68.03 |
| Epoch: 14 | Train Loss: 0.800 | Train Acc: 65.14 | Val. Loss: 0.824 | Val. Acc: 65.99 | Test Loss: 0.790 | Test Acc: 66.78 |
| Epoch: 15 | Train Loss: 0.784 | Train Acc: 66.08 | Val. Loss: 0.829 | Val. Acc: 64.26 | Test Loss: 0.788 | Test Acc: 65.45 |
| Epoch: 16 | Train Loss: 0.780 | Train Acc: 66.69 | Val. Loss: 0.823 | Val. Acc: 64.43 | Test Loss: 0.790 | Test Acc: 66.33 |
| Epoch: 17 | Train Loss: 0.784 | Train Acc: 65.80 | Val. Loss: 0.789 | Val. Acc: 66.34 | Test Loss: 0.762 | Test Acc: 68.06 |
| Epoch: 18 | Train Loss: 0.761 | Train Acc: 67.23 | Val. Loss: 0.803 | Val. Acc: 66.69 | Test Loss: 0.769 | Test Acc: 67.32 |
| Epoch: 19 | Train Loss: 0.743 | Train Acc: 67.96 | Val. Loss: 0.810 | Val. Acc: 66.34 | Test Loss: 0.766 | Test Acc: 67.90 |
| Epoch: 20 | Train Loss: 0.735 | Train Acc: 68.62 | Val. Loss: 0.820 | Val. Acc: 66.60 | Test Loss: 0.773 | Test Acc: 67.22 |
| Test Loss: 0.773 | Test Acc: 67.22 |
This film is terrible negative
This film is great positive
this film is great positive
this film is good positive
this film is bad negative
This film is not bad negative
My friend likes awesome food positive
My friend likes awful recipes negative
the film is amazingly delightful to watch positive
the film is boring negative
the film is not good negative
the film is fun positive
the film is awful negative
the film is bad negative
the film is a true story positive
the film is a fake story positive
i like this phone negative
i hate this phone negative
the camera is priceless positive
the camera is expensive positive
Sentence : This film is terrible
              0      1      2         3
Sentence   This   film     is  terrible
IntegGrad     0  0.134  0.218     0.848
PREDICTED Label : negative
Sentence : This film is great
              0     1      2      3
Sentence   This  film     is  great
IntegGrad     0  0.06  0.055  0.253
PREDICTED Label : positive
Sentence : this film is great
               0      1      2      3
Sentence    this   film     is  great
IntegGrad -0.066  0.148  0.065  0.225
PREDICTED Label : positive
Sentence : this film is good
               0      1      2     3
Sentence    this   film     is  good
IntegGrad -0.081  0.133  0.036  0.44
PREDICTED Label : positive
Sentence : this film is bad
               0      1      2      3
Sentence    this   film     is    bad
IntegGrad  0.038  0.059  0.158  0.932
PREDICTED Label : negative
Sentence : This film is not bad
              0      1      2      3     4
Sentence   This   film     is    not   bad
IntegGrad     0  0.094  0.132  0.076  0.71
PREDICTED Label : negative
Sentence : My friend likes awesome food
            0       1      2        3      4
Sentence   My  friend  likes  awesome   food
IntegGrad   0  -0.099 -0.044    1.125  0.001
PREDICTED Label : positive
Sentence : My friend likes awful recipes
            0       1      2      3        4
Sentence   My  friend  likes  awful  recipes
IntegGrad   0  -0.068  0.061  0.698        0
PREDICTED Label : negative
Sentence : the film is amazingly delightful to watch
               0      1      2          3           4      5      6
Sentence     the   film     is  amazingly  delightful     to  watch
IntegGrad -0.106  0.008 -0.035      0.691       1.011 -0.077  0.094
PREDICTED Label : positive
Sentence : the film is boring
               0      1      2       3
Sentence     the   film     is  boring
IntegGrad -0.063  0.094  0.265   0.737
PREDICTED Label : negative
Sentence : the film is not good
               0      1      2      3      4
Sentence     the   film     is    not   good
IntegGrad -0.009  0.103  0.166  0.468 -0.179
PREDICTED Label : negative
Sentence : the film is fun
              0      1     2      3
Sentence    the   film    is    fun
IntegGrad -0.09  0.116  0.04  0.888
PREDICTED Label : positive
Sentence : the film is awful
               0      1      2      3
Sentence     the   film     is  awful
IntegGrad -0.026  0.134  0.265  0.662
PREDICTED Label : negative
Sentence : the film is bad
               0      1      2      3
Sentence     the   film     is    bad
IntegGrad -0.039  0.093  0.166  0.988
PREDICTED Label : negative
Sentence : the film is a true story
               0      1      2     3      4      5
Sentence     the   film     is     a   true  story
IntegGrad -0.028  0.168  0.064  0.11  0.195  0.171
PREDICTED Label : positive
Sentence : the film is a fake story
               0     1     2      3      4      5
Sentence     the  film    is      a   fake  story
IntegGrad  0.057  0.18  0.05  0.105 -0.187  0.093
PREDICTED Label : positive
Sentence : i like this phone
               0      1      2      3
Sentence       i   like   this  phone
IntegGrad -0.113  0.093  0.079  0.096
PREDICTED Label : negative
Sentence : i hate this phone
               0      1      2      3
Sentence       i   hate   this  phone
IntegGrad -0.019  0.115  0.117  0.119
PREDICTED Label : negative
Sentence : the camera is priceless
               0       1      2          3
Sentence     the  camera     is  priceless
IntegGrad -0.062  -0.007  0.037      0.856
PREDICTED Label : positive
Sentence : the camera is expensive
               0       1      2          3
Sentence     the  camera     is  expensive
IntegGrad  0.014   0.018  0.009      0.196
PREDICTED Label : positive
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
