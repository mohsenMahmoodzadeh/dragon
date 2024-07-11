# DRAGON: a Dedicated RAG for October 7th News conflict

[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)]()
[![Conference](http://img.shields.io/badge/conference-ACL--2022-4b44ce.svg)]()

## 🛠 Setup
Run `init.sh` to install the dependencies of the project:

```
sh init.sh
conda activate venv
```

## 🏃‍♂️ Running Experiments

### Preprocessing and generating train/test splits 
To apply the initial preprocessing of the annotated data, run the following commands, which will gives the train and 
test splits for the following tasks, respectively:

- Bias (`Bias`)
- Propaganda (`Propaganda`)

```
python preprocess.py --task="Propaganda" 
```

### Embedding generation
Run the following command to get the train/test embeddings for the specified embedding model:

- [Multilingual E5 Large](https://huggingface.co/intfloat/multilingual-e5-large): `ML-E5-large` 
- [BGE M3](https://huggingface.co/BAAI/bge-m3): `BGE-M3` 
- [E5 Mistral 7B](https://huggingface.co/intfloat/e5-mistral-7b-instruct): `E5-mistral-7b` 
- [Nomic Embed](https://huggingface.co/nomic-ai/nomic-embed-text-v1): `Nomic-Embed`


```
python generate_embeddings.py --task="Propaganda" --embedding_method="ML-E5-large"
```

### Classification
Run the following command to employ the generated embeddings for training and testing with the following classification 
methods:

- KNN (`knn`)
- SVC (`svc`)
- Xgboost (`xgboost`)

```
python experiment_classification.py --task="Propaganda" --embedding_method="ML-E5-large --cls_method="knn"
```








## 🤝 Authors
- Sadegh Jafari: [[Linkedin](https://www.linkedin.com/in/sadegh-jafari-b2a55b229)] - [[Google Scholar](https://scholar.google.com/citations?user=hgopDk0AAAAJ&hl=en)]
- Mohsen Mahmoodzadeh: [[Linkedin](https://ir.linkedin.com/in/mohsen-mahmoodzadeh)] - [[Google Scholar](scholar.google.com)]
- Vanooshe Nazari: [[Linkedin](https://ir.linkedin.com/in/vanooshe-nazari-b98476276)] - [[Google Scholar](scholar.google.com)]
- Razieh Bahmanyar: [[Linkedin](https://www.linkedin.com/in/shahrzad-bahmanyar/)] - [[Google Scholar]()]
- Kate Burrows: [[Linkedin](https://www.linkedin.com/in/kate-burrows-ph-d/)] - [[Google Scholar](https://scholar.google.com/citations?user=Z3GFplAAAAAJ&hl=en)]

## 📖 Citing Dragon

If you use DRAGON in your research, please consider citing the paper as follows::

```
@inproceedings{jafari-etal-2024-dragon,
    title      = "DRAGON: a Dedicated RAG for October 7th conflict News",
    author     = "Jafari, Sadegh and Mahmoodzadeh, Mohsen and Nazari, Vanooshe and Bahmanyar, Razieh, and Burrows, Kathryn",
    booktitle  = "Proceedings of the 2nd ArabicNLP Annual Meeting of the Association for Computational Linguistics (ACL 2024)",
    month      = july,
    year       = "2024",
    address    = "Bangkok, Thailand",
    publisher  = "Association for Computational Linguistics"
}
```