# ROP299
Research Opportunity Program: Summarizing Learners’ Free Responses at Large Scales

## Preparation

### 1. Install PyCharm
First, you may need to downlaod and install [PyCharm](https://www.jetbrains.com/pycharm/).

### 2. Download the project
Download the project and open it in the PyCharm.

### 3. Install packages
You can install packages using or without using `pip`.
#### 3.1 Using `pip`
Run following commands in your PyCharm terminal to install packages:

Install [pandas](https://pandas.pydata.org/docs/)
```bash
pip install pandas
```

Install [nltk](https://www.nltk.org/)
```bash
pip install nltk
```

#### 3.2 Without using `pip`
- In the toolbar, click `PyCharm -> Preferences -> Project: ROP -> Python Interpreter -> "+"`
- Enter the package name and click `Install Package`


## Usage

### 1. Preprocessing
Create a new folder in `GloVe` folder to store the preprocessing results.\
\
**Note:** If you want to compare different results, you can either 
  - choose a descriptive name for this folder or 
  - Use `resultx` (like `result1` in `GloVe`) and take notes of it

In this new folder, create an empty txt file named `survey.txt`

#### 1.1 generate the txt file
- Open `preprocessing.py`
- If you want to remove stop words, uncomment `line 28` and `line 29` 
- In `line 41`, change `'GloVe/result1/survey.txt'` to `'GloVe/resultx/survey.txt'`
- In `line 41`, change `[8]` to the list of column numbers that you want use

| column number | Question |
| --- | --- |
| 8 | Why is this your preferred mode?  |
| 18 | If you could change one thing about the way your online classes are designed, what would you change? Why? |
| 25 | If you could change one thing about the way your in-person classes are designed, what would you change? Why?  |
| 27 | Why do you prefer online or in person courses?  |

- Run `preprocessing.py`, and you will see the generated text in `GloVe/resultx/survey.txt`.

#### 1.2 word embeddings
- Open `GloVe/demo.sh`
- In `line 18 19 20 21 23`, change `result1` to `resultx`
- Change the parameters in `line 24` ~ `line 32`
- Run `demo.sh` (click the button near `line 1`), and you will see the generated txt files in `GloVe/resultx`. 
- The vector representation for each word is in `vectors.txt`.

