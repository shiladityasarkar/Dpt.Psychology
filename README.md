# Dpt.Psychology
## by Shiladitya Sarkar
 _project for the Department of Psychology._
 
 ### Dataset:
 ~~confidential~~
 
 ### How to run (authorized users only):
 clone this git repo and then modify `Generate('path_to_the_dataset')` in `run.py`. Just hit run.

 ### What will happen?
 Requirements will be automatically installed. To disable status OKs upon repeated runs just comment out `shila.requirements()` and `shila.download()` from the `__init__()` of `clusterAI`.

 If you want to manually care about installations for some reason, please check out `requirements.txt`.
 
 Directories along with files will be automatically created according to the following structure under root -

 ```bash
 |
 |_wordclouds
   |__cluster3
      |___..png
   |__cluster4
   |__...
 |
 |_clustered_responses
   |__cluster3
      |___..txt
   |__cluster4
   |__...
```
### How to read?
the naming convensions are : `ci_j` means the $j^{th}$ cluster items in $i$ number of clusters where $j<=i$.

the extension convernsions are : `.txt` contains the clustered responses and `.png` contains the word clouds.

### Also,
Each response file contains **metadata** which holds overall information about the cluster and individual reports for each candidate's response which equivalents to a single point in that cluster.

### Update
this project now supports both excel and csv files (`.xlsx`, `.csv`).

To request support for more file types please contact the author at writetoshiladitya@gmail.com
