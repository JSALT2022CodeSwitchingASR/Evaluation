### Installation
1. Install libraries:

```
pip install epitran
pip install abydos
pip install camel-tools
pip install pyenchant
pip install nltk
```
2. Install nltk resources in python environment:
```
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
```
3. If you are on the QCRI cluster set the following:
```
export PATH=$PATH:/export/sharedapps/flite/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/export/sharedapps/flite/build/x86_64-linux-gnu/lib:/export/sharedapps/flite/include
```

### Testing the script 
`python psd.py hyp.txt ref.txt`

The script should generate results.txt

```
ID: 1
REF: a kind of
HYP: ا كايند اوف
REF phone: əkajndʌv
HYP phone: akaindauf
PER: 0.625 PSD: 0.25012 PSD_norm: 0.213

ID: 2
REF: لا في at least a chance نحاول مرة extra معاه يمكن يبدأ يبقى more flexible
HYP: لا في atlista chance نحاول مرة extra معي يمكن يبدأ يبقى more flexible
REF phone: lafiætlistəʧænsnħaulmrtɛkstɹəmʕahimknibdaibqamɔɹflɛksəbəl
HYP phone: lafiʧænsnħaulmrtɛkstɹəmʕiimknibdaibqamɔɹflɛksəbəl
PER: 0.15789 PSD: 0.14488 PSD_norm: 0.13514
```

