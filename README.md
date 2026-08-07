# SomOffXplain

This repository is the official implementation of the [Scientific Reports](https://www.nature.com/srep/)  **May-2026** Paper with the title: [Benchmarking explainable offensive language detection in Somali with human-annotated rationales](https://www.nature.com/articles/s41598-026-53781-0) 



# Overview

Offensive language presents significant challenges on the internet and requires robust moderation. However, the efficacy of such moderation often depends on providing clear and interpretable justifications for each classification. Unfortunately, many
existing datasets lack annotated rationales, and most detection models offer limited interpretability and transparency. These limitations hinder the development of trustworthy systems and the implementation of effective content moderation strategies. In this paper, we introduce SomOffXplain, an interpretable framework for detecting offensive language in Somali, which generates
human-understandable explanations for its predictions. SomOffXplain performs span-level rationale extraction at both the word and phrase levels, enabling it to highlight text segments that support its predictions. Given that Somali is a low-resource
language, we first construct a new benchmark dataset of 10,175 samples, each annotated with human-provided rationales. We evaluate our method against five fine-tuned pre-trained models using Local Interpretable Model-Agnostic Explanations (LIME). Additionally, we adapt four large language models (LLMs) through few-shot and zero-shot prompting to assess their ability to
understand and produce rationales in Somali. Our proposed model demonstrates superiority in terms of explainability and predictive accuracy, exhibiting higher plausibility and faithfulness compared to the baselines. Furthermore, our results reveal that half of the state-of-the-art LLMs evaluated fail to generate high-quality rationales that align with human-annotated ground truth rationales, whereas LIME-based methods also prove to be weak explainers for Somali text. We believe our contributions support online safety, help prevent harassment in under-resourced language communities, enhance the trustworthiness of language models, and promote transparency in artificial intelligence systems.

---------------------------------------------------------------------------------------------

## Notes


 📋Due to the sensitive nature of the data, the XOLDS-dataset is available upon reasonable request and in line with responsible NLP data-sharing practices. To request access, click **Issues** in the top-left corner of this repository and complete the request form. For further information, please contact **Abdisalam** **Badel** at fiicane121@gmail.com or 202214090105@std.uestc.edu.cn.

```
NaN in the target column of the data means there is no target; in other words, the text is positive (not offensive).
```

## Usage 


#### 1. Requirements
 ```
conda == 24.11.0
torch == 2.5.1
transformers == 4.46.3
```

#### 2. Training

📋 To retrain this model, simply run the following command.

 ```
cd Code
python main.py train --train_csv data/train.csv --dev_csv data/dev.csv --epochs 3 --batch_size 8 --save_path best_model.pth 
```

#### 3. Evaluation 

```
cd Code
python main.py eval --test_csv data/test.csv --save_path best_model.pth --plot
```

#### 4. Inference 

```
cd Code
python main.py explain --input_text "Intaad munaafiq tahay badanaa"
```




***

# 	Annotator Guidelines Used for Data Annotation
We have shared the following guidelines with our annotators and held video conversations with them via Zoom. In addition, we discussed many issues through WhatsApp calls. Our contact was on a daily basis throughout the data annotation process. Please note that in the guideline, the English text is not a direct translation of the Somali text. In cases where we use direct translation, we label it as “English Translation.”
##  Label and Target Annotation Guidelines
**Somali.** Shuruucda, la raacayo si loo anatoydgareeyo dhaatadan, fadlan fiiri oo raac qeexitaanka ka hore intaadan, bilaabin shaqada, Mahadsanid.

**English.** “Annotation instructions for the dataset: please refer to these definitions before annotating the data, thanks.”

**Offensive definition: Somali.** Waa wax walba oo ka xanaajiya ama ay dhibsadaan, dad gooni ah. Waa sida cayda, aflagaadada, waxyeelada, ama ficilo midib takoor ah. Kuwaas oo ku salaysan qabiil, diin, midib, ama asalka koox ama ruux leeyahay.

**English.** “Offensive language means any utterance that makes someone or a group feel upset or annoyed. It may include insult, harmful, or discriminatory actions against people based on ethnicity, race, color, gender, religion, nationality, etc. In short, offensive language is any improper speech that breaks the accepted standard of everyday speech, whether intended to target someone or not.”

**Not-offensive: Somali.** Qoraal caadi ah, waa qoraalka aan lahayn wax xanaf ah, ama aan cay kujirin. 

**English.** “Any text that does not contain insults, derogatory language, or profanity and is accepted by the community’s norms and traditions is considered normal or not offensive.”

  **Person: Somali.** Qoraalka shaqsi ataaga ah, waa marka faaladu ay weerar ku tahay ruux, kaas oo la magacaabay, ama la xiganayo.
  
 **English.** “Select a comment as a person if it is directly targeted towards an individual. It means the person can be named indirectly, mentioned, or referenced. It is an offensive just about a person.”
 
**Group: Somali.** Cayda koox, waa marka koox si wada jir ah loo weerarayo iyadoo la adeegsanayo qoraal ama faalo. Marmar waxay noqon kartaa marka la ataag gareeyo koox iyada oo lagu xumaynayo aragtidooda siyaasadeed, ama asalka Meesha ay kazoo jeedaan. 

**English.** “Offensive comments can be classified as a group if they target a group of people collectively. An example of this type of category can be when the comment is towards a group of people based on their political view, geographical location, sub-group, etc.”

 **Woman: Somali.** Faalada dumarka, faaladani waxay noqonkartaa mid dumarka wax looga sheegayo waa marka faalooduhu uu qof dumar ah toos u magacaabo. Isaga oo adeegsanaya weedhaha, iyada, naagtan, ama erayo cay ah oo dumarka u gooni ah sida dhilo ama sharmuuto.
 
 **English.** “The offensive falls in this category when the comment is specific to women, for example, when a character only women own is mentioned in the comment. It can be sex-related terms or other terms such as her, this woman, etc. It is different from person-related comments. In-person comments we do not have specific terms for women; they are just about a person, maybe a man or woman.”
 
**Other: Somali.** Faalada other, waa cayda aan cidna taargad garaynaynin ee laakiin caay ama wax dhimaal ah waxaan dhahaynaa ”Other” oo macnaheedu yahay. Waa cay aan cidna abaaraynin. 

**English.** “In this case, the offensive is general; it does not target an individual or group. It is offensive but not towards anyone; it does not mention either name, person, or group.”

**Tusaale/Example.** Moryankan sheydanka miyeey kadhigen masul wa kuwi xasuqay kow kayahay wana Dambile shacab: **Offensive (Person)**.

**English Translation.** “Did they make this thief, Satan, an official, the number one among those who massacred. He is a civil criminal.”

##  Rationale Selection Guidelines

**Somali.** Fadlan dooro qaybta weedhahan ka mida ah ee ka dhigaysa kuwo xanaf leh ama kuwo caadi ah. Tusaale weedha soo socoto waxaan uga dhignay tu xanaf leh sababtoo ah erayada madaw ah ayaa xanaf leh. Fadlan weedh walba dooro rationale sida midan oo kale.

**English Translation.** “Please select the part of the text that makes the sentences offensive or non-offensive. For example, the following sentence is considered offensive because the bolded words are offensive words. Please select the rationale words for each sentence in the data in the same way as in this example.”

Weedha/sentence: **Moryankan** **sheydanka** miyeey kadhigen masul wa kuwi **xasuqay** kow kayahay wana **Dambile** shacab.


## 📌 Citation

If you find this repository helpful, please cite our paper:


```
@article{Badel2026,
    author = {Badel, A. M. and Zhong, T. and Xu, X. et al.},
    title = {Benchmarking explainable offensive language detection in {Somali} with human-annotated rationales},
    journal = {Sci Rep},
    volume = {16},
    pages = {24406},
    year = {2026},
    doi = {10.1038/s41598-026-53781-0}
}
```

License
---------------------------------------------------------------------------------------------------------
MIT


Contact 
---------------------------------------------------------------------------------------------------------
For any inquiries, please contact **Abdisalam** **Badel** at fiicane121@gmail.com or 202214090105@std.uestc.edu.cn.
