# Phoneme similarity edit distance

This is an emplementation of the edit distance in the phoneme (IPA) space with the substitusion weight scaled by the (1-similarty) of the phonemes. The script reports three metrics: 
- PER: similar to CER but measured in the phoneme space
- PSD: same as PER but the substitusion weight scaled by the (1-similarty) between phonemes.
- PSD_norm: PSD but after removing the vowels 

### Installation
1. Install libraries:

```
pip install epitran
pip install abydos
pip install camel-tools
apt install -qq enchant
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
3. If you are on the QCRI cluster source the following for flite:
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
REF phone: ə kajnd ʌv
HYP phone: a kaind auf
PER: 0.625 PSD: 0.375 PSD_norm: 0.2258

ID: 2
REF: لا في at least a chance نحاول مرة extra معاه يمكن يبدأ يبقى more flexible
HYP: لا في atlista chance نحاول مرة extra معي يمكن يبدأ يبقى more flexible
REF phone: la fi æt list ə ʧæns nħaul mrt ɛkstɹə mʕah imkn ibda ibqa mɔɹ flɛksəbəl
HYP phone: la fi ætlɪstə ʧæns nħaul mrt ɛkstɹə mʕi imkn ibda ibqa mɔɹ flɛksəbəl
PER: 0.05263 PSD: 0.03112 PSD_norm: 0.02703

ID: 3
REF: artificial
HYP: ارتficial
REF phone: ɑɹtəfɪʃəl
HYP phone: art fɪʃəl
PER: 0.33333 PSD: 0.16844 PSD_norm: 0.0516



Mean PER_tot:     0.14865
 Mean PSD_tot:      0.085
 Mean PSD_norm_tot: 0.05079
```

4. To run the pipeline 

`./run.sh data/dir`
