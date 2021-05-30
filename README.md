# Criticize - Machine Learning

Machine learning aspect of the project.

## About The Project

This project - `Criticize` - aims to complement humans’ critical thinking abilities by harnessing machine learning algorithms with natural language processing (NLP) techniques. This algorithm will be able to read an article, and generate critical thinking questions based on the content or context of the article.

### Built With

* [Simple Transformers](https://simpletransformers.ai/) - A T5 Model that can be used to predict different tasks.

## Getting Started

### Prerequisites

* [Python3 and pip](https://www.python.org/downloads/)


### Installation

1. Retrieve the files

```
git clone https://github.com/sebbycake/criticize-ML.git
```

2. Change directory
```
cd criticize-ML
```

3. Create virtual environment. Note that I was using `virtualenvwrapper` as of this writing.
```
mkvirtualenv criticize 
```

4. Activate virtual environment (it will automatically activate when you first create it)
```
workon criticize
```

5. Install Python packages
```
pip install -r requirements.txt
```

### Usage
```
# To train the model:
python train.py

# To generate predictions:
python generate.py
```

Credits: [Asking the Right Questions: Training a T5 Transformer Model on a New task](https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c)

Read the documentaton: [T5 Model - Simple Transformers](https://simpletransformers.ai/docs/t5-model/)

## License

This project is licensed under the MIT License.
