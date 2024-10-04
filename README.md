# DRAGON: a Dedicated RAG for October 7th News conflict

[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://aclanthology.org/2024.arabicnlp-1.58/)
[![Conference](http://img.shields.io/badge/conference-ACL--2024-4b44ce.svg)](https://2024.aclweb.org/)

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

### Retrieval Augmented Generation (RAG)
Run the following command to employ multilingual embeddings and OpenAI LLM to get inference:
```
python rag.py --task="Propaganda" --k=7 --embedding_method="intfloat/multilingual-e5-base"
```

**Note**: You should have an OpenAI API key to generate responses from input prompts. Put your API key in a `.env` file
in the root directory of the project and set it as shown below:
```
OPENAI_API_KEY=<YOUR_KEY>
```

### IAA Prediction
Run the following command to employ the generated embeddings and classification methods to get inferences for IAA 
(Inter-Annotator agreement) data:

```
python inference_iaa.py --task="Propaganda" --embedding_method="ML-E5-large --cls_method="knn" --iaa_index=1
```


## 🤝 Authors
- Sadegh Jafari: [Linkedin](https://www.linkedin.com/in/sadegh-jafari-b2a55b229) - [Google Scholar](https://scholar.google.com/citations?user=mcJ6RoUAAAAJ&hl=en)
- Mohsen Mahmoodzadeh: [Linkedin](https://ir.linkedin.com/in/mohsen-mahmoodzadeh) - [Google Scholar](https://scholar.google.com/citations?hl=en&user=0bJEyegAAAAJ)
- Vanooshe Nazari: [[Linkedin](https://ir.linkedin.com/in/vanooshe-nazari-b98476276) - [Google Scholar](https://scholar.google.com/citations?user=m4r-eNkAAAAJ&hl=en)
- Razieh Bahmanyar: [[Linkedin](https://www.linkedin.com/in/shahrzad-bahmanyar/) - [ResearchGate](https://www.researchgate.net/profile/Razieh-Bahmanyar)
- Kate Burrows: [[Linkedin](https://www.linkedin.com/in/kate-burrows-ph-d/) - [Google Scholar](https://scholar.google.com/citations?user=Z3GFplAAAAAJ&hl=en)

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