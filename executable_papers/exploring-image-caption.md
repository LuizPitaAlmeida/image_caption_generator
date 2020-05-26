# Exploring image caption generator using Convolutional Neural Networks and moderns Natural Language processing models

![Pytorch_logo.png][nextjournal#file#99d957af-fae0-41ea-afab-949696fd3c60]

## Abstract

## GitHub

[github-repository][nextjournal#github-repository#d9732e58-bbf7-40c8-81e0-60e79db38101]

## Introduction

With de emergence of deep learning, specially the advent of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), the image captioning area, like many other subjects of Computer Vision, achieved new state of art results. The image captioning is an automatic way to describe an image, generating a caption for the scene. It can help in many applications, such as image retrieval. The image captioning incorporate many areas of Computer Vision, such as object detection, object recognition, scene understanding, object properties and their interactions. All of these are necessary to make a machine understand an image, but  it also need to learn how to generate a sentence. \[Survey\]

Another research area that got an upgrade with the advances on deep learning is the Natural Language Processing (NLP). This area gotten better results with the use of neural models, followed by the use of RNN, such as LSTM (Long Short Term Memory) and GRU (Gated Recurrent Unit). Nowadays, the state of arts results are focus on the use of attention mechanisms, mainly with the advent of Transform models. Pre-trained Transform models like BERT (Bidirectional Encoder Representations from Transformers) and T5 are most modern state of art methods in NLP. \[Maybe attention is all you need (review), and BERT and T5\]

In general, image caption models follow a encoder-decoder architecture, where the encoder is an image feature extractor, mostly a CNN, that could be associated to a language model encoder to generate a jointly embedded representation of words and images. This association forms the multi-modal language models. On the other hand, the decoder is a text generator model, mainly using LSTM language models, that translate the embedding from encoder into a sentence. This approach is mainly trained using a supervised method. Others approaches include the usage of reinforcement learning, unsupervised learning, attention mechanisms, semantic concepts, additional code blocks that check the quality of the text generated, generative adversarial networks, and others. \[Survey\]

This paper focus in implement a image caption generator demonstration that follows a encoder-decoder architecture. The encoder is a CNN model only to extract images features, so it is not a multi-modal encoder. The decoder uses a modern NLP pre-trained model, the BERT, to generate the captions.  

## Related works

Show and Tell (2015)

Show, Attend and Tell (2015)

BERT (2019)

## Materials and Methods

In this section we describe the materials such as data and programming language used, and adopted methodology.

### Programing Language

For this project we are using the Python 3 language with the Pytorch 1.3 library. Pytorch is an open source machine learning framework designed to accelerate research prototyping. In the last years together with TensorFlow is the most used framework for deep learning.

```python id=2cc038ea-4dad-4cd3-b255-e4fe2c4cc404
import platform, torch
print("Python version: %s.\nPyTorch version: %s." %
      (platform.python_version(),torch.__version__))
```

### MS COCO database$$

The Microsoft Common Objects in Context is a large and well knowing database used for benchmark many Computer Vision applications.  It has a dataset for image captioning containing more than 330000 images with five different descriptions for each one. 

Let's download and unzip the data. Data download links could be find in:

[download_links.txt][nextjournal#file#71ddd10c-207a-4473-a709-8e43c3eb46d2]

To download we used a bash cell with `wget` command.

```bash id=150585be-fcca-4821-a1da-b91a650f258a
wget --progress=dot:giga -P /results -i [reference][nextjournal#reference#c52b7b53-f891-40b3-b443-eba915e57b98]
```

[test2014.zip][nextjournal#output#150585be-fcca-4821-a1da-b91a650f258a#test2014.zip]

[train2014.zip][nextjournal#output#150585be-fcca-4821-a1da-b91a650f258a#train2014.zip]

[val2014.zip][nextjournal#output#150585be-fcca-4821-a1da-b91a650f258a#val2014.zip]

[image_info_test2014.zip][nextjournal#output#150585be-fcca-4821-a1da-b91a650f258a#image_info_test2014.zip]

[annotations_trainval2014.zip][nextjournal#output#150585be-fcca-4821-a1da-b91a650f258a#annotations_trainval2014.zip]

We had download the above files, and now let's unzip only the validation files for a simple demonstration.

```bash id=933c43f5-42db-4187-81de-b96ec6ce3a86
unzip [reference][nextjournal#reference#ad5390a5-61d9-4750-9c4d-1f4b22449c91]
unzip [reference][nextjournal#reference#a0c174eb-4650-429b-ad83-d292728d56ac]
# unzip [reference][nextjournal#reference#6d2afe31-01b9-4363-8b6a-591ebbc22e22]
# unzip [reference][nextjournal#reference#d3508725-8cdf-416b-9ca3-8f97cca1a142]
```

Validation annotations were saved in `annotations/captions_val2014.json` and the images in `val2014/`. Let's open it using torch vision dataset tool. But first we need to install COCO python API.

```bash id=8c1ea37e-ca3e-4f9e-b718-cb31d20e8d55
conda install -c conda-forge pycocotools
```

```python id=aaaed38c-05c0-4ad8-9d5a-f1e42eb52dd8
import torchvision 
from torchvision import datasets, models, transforms

test_data = datasets.CocoCaptions(
  root='val2014',
  annFile='annotations/captions_val2014.json',
  transform=transforms.ToTensor())

print('Number of samples: ', len(test_data))
```

Let's look for a sample and its labels. In the image we selected only the first label for a better visualization.

```python id=a2a64983-107b-43ec-9a66-79e88e1b533b
from matplotlib import pyplot as plt

img, target = test_data[4] # load 5th sample

print("Image Size:", img.size())
print("Labels:")
for label in target:
	print(label)

plt.title(target[0])
plt.imshow(img.permute(1, 2, 0))
plt.gcf()
```

![result][nextjournal#output#a2a64983-107b-43ec-9a66-79e88e1b533b#result]

### Convolutional Neural Networks

We decided to use a common used CNN for image feature extraction, in this case, we chosen the ResNet-101.

```python id=8697ac42-8bc8-4586-afaf-a41829beb77c
import torchvision.models as models
resnet101 = models.resnet101(pretrained=True)
```

```python id=f14263af-258c-4c75-bfa9-d8334becc3a0
print(resnet101)
```

### BERT - Bidirectional Encoder Representations from Transformers

BERT is recent paper published by Google AI team that, in the last year, got state-of-art results in a wide range of NLP tasks. It is designed using only attention mechanism, Transformer module \[attention is all you need\] and considering a bidirectional context.

Bert was trained to be a language model that understand better the language context and flow. Researches notice that BERT could be distribute as a pre-trained model such as pre-trained CNNs,  so common in Computer Vision area.

```bash id=b531084f-b926-4cba-b7a4-3f5e92438b29
pip install transformers
```

```python id=78d7615a-47aa-4287-a958-e662e7f681fe
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
print(bert)
```

### Our encode-decoder model

In this section we present our encoder-decoder model. The encoder will be formed by the feature extraction of ResNet-101, while in the decoder will use the BERT to generate contextualized word vectors that are after detokenized and the predicted captions are formed using BERT embedding and images features together in a attention mechanism module. This module is based in <<https://github.com/ajamjoom/Image-Captions>> implementation.

WITHOUT TRAINING vs SUPERVISED

CPU/GPU

METRICS

## Experiments and results

From experiments we expect to compare our results with related work results.

## Discussion and Conclusions

## References

[nextjournal#file#99d957af-fae0-41ea-afab-949696fd3c60]:
<https://nextjournal.com/data/Qmf3UDWgi8zVgPR6Mo6ebVgzSpufsiELMRWdu2aa8nysTb?filename=Pytorch_logo.png&content-type=image/png> (Vinyals et al - Show and Tell: A Neural Image Caption Generator \(2015\).)

[nextjournal#github-repository#d9732e58-bbf7-40c8-81e0-60e79db38101]:
<https://github.com/LuizPitaAlmeida/image_caption_generator>

[nextjournal#file#71ddd10c-207a-4473-a709-8e43c3eb46d2]:
<https://nextjournal.com/data/QmbGGMwq9367tBsBRwFjbuHjUj5qDQcqTQQST41nMf7JYi?filename=download_links.txt&content-type=text/plain>

[nextjournal#reference#c52b7b53-f891-40b3-b443-eba915e57b98]:
<#nextjournal#reference#c52b7b53-f891-40b3-b443-eba915e57b98>

[nextjournal#output#150585be-fcca-4821-a1da-b91a650f258a#test2014.zip]:
<https://nextjournal.com/data/Qme9KeWLidXoPmduo2TGHMGfQtfCefn4sy6KE5TwbqtrZ4?filename=test2014.zip&content-type=application/zip>

[nextjournal#output#150585be-fcca-4821-a1da-b91a650f258a#train2014.zip]:
<https://nextjournal.com/data/QmeMH1Pqjr7Q2MjZCkz1j8sd8HHFpoPDqTAiTLSwYNw6Rr?filename=train2014.zip&content-type=application/zip>

[nextjournal#output#150585be-fcca-4821-a1da-b91a650f258a#val2014.zip]:
<https://nextjournal.com/data/QmfUY9ceVUXAshr5KpbkBzgjWe1KLhSXZgvtu8THbJr5BM?filename=val2014.zip&content-type=application/zip>

[nextjournal#output#150585be-fcca-4821-a1da-b91a650f258a#image_info_test2014.zip]:
<https://nextjournal.com/data/QmWkJfmR5ehN6rVwvXMFDqv9p1NDDTYVFLJ9e6xMrUZoRJ?filename=image_info_test2014.zip&content-type=application/zip>

[nextjournal#output#150585be-fcca-4821-a1da-b91a650f258a#annotations_trainval2014.zip]:
<https://nextjournal.com/data/QmNYeNZtarkaCjdmYAribzgQfNMyc3WnATZuswjPxFa7hz?filename=annotations_trainval2014.zip&content-type=application/zip>

[nextjournal#reference#ad5390a5-61d9-4750-9c4d-1f4b22449c91]:
<#nextjournal#reference#ad5390a5-61d9-4750-9c4d-1f4b22449c91>

[nextjournal#reference#a0c174eb-4650-429b-ad83-d292728d56ac]:
<#nextjournal#reference#a0c174eb-4650-429b-ad83-d292728d56ac>

[nextjournal#reference#6d2afe31-01b9-4363-8b6a-591ebbc22e22]:
<#nextjournal#reference#6d2afe31-01b9-4363-8b6a-591ebbc22e22>

[nextjournal#reference#d3508725-8cdf-416b-9ca3-8f97cca1a142]:
<#nextjournal#reference#d3508725-8cdf-416b-9ca3-8f97cca1a142>

[nextjournal#output#a2a64983-107b-43ec-9a66-79e88e1b533b#result]:
<https://nextjournal.com/data/QmZsqhSmCKSPtnVkgTDTz7TxhwkMLwQN4miZwz2BsAku5j?content-type=image/svg%2Bxml>

<details id="com.nextjournal.article">
<summary>This notebook was exported from <a href="https://nextjournal.com/a/MaQo4ptSYDSC7jGKsoSR1?change-id=ChyPwx6EF8Uz9YLSsTTKsa">https://nextjournal.com/a/MaQo4ptSYDSC7jGKsoSR1?change-id=ChyPwx6EF8Uz9YLSsTTKsa</a></summary>

```edn nextjournal-metadata
{:article
 {:settings nil,
  :nodes
  {"150585be-fcca-4821-a1da-b91a650f258a"
   {:compute-ref #uuid "b3f96275-e5ba-4896-be82-e74f77d81b8d",
    :exec-duration 742826,
    :id "150585be-fcca-4821-a1da-b91a650f258a",
    :kind "code",
    :locked? true,
    :output-log-lines {:stdout 865},
    :runtime [:runtime "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"],
    :stdout-collapsed? false},
   "2cc038ea-4dad-4cd3-b255-e4fe2c4cc404"
   {:compute-ref #uuid "be0da0bd-0661-4e5f-8f97-b2d2285a5b7e",
    :exec-duration 925,
    :id "2cc038ea-4dad-4cd3-b255-e4fe2c4cc404",
    :kind "code",
    :output-log-lines {:stdout 3},
    :refs (),
    :runtime [:runtime "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"]},
   "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"
   {:environment
    [:environment
     {:article/nextjournal.id
      #uuid "5b5615a1-6b2b-4cef-9620-9858a4f0f2f3",
      :change/nextjournal.id
      #uuid "5df706bf-9a09-43e3-8de2-ff9647fb5215",
      :node/id "24f5f730-f1c8-497a-a1e7-b5b623450b49"}],
    :environment? true,
    :id "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325",
    :kind "runtime",
    :language "python",
    :name "PyTorch",
    :type :nextjournal,
    :runtime/mounts
    [{:src [:node "d9732e58-bbf7-40c8-81e0-60e79db38101"],
      :dest "/image_caption_generator"}]},
   "6d2afe31-01b9-4363-8b6a-591ebbc22e22"
   {:id "6d2afe31-01b9-4363-8b6a-591ebbc22e22",
    :kind "reference",
    :link
    [:output
     "150585be-fcca-4821-a1da-b91a650f258a"
     "image_info_test2014.zip"]},
   "71ddd10c-207a-4473-a709-8e43c3eb46d2"
   {:id "71ddd10c-207a-4473-a709-8e43c3eb46d2", :kind "file"},
   "78d7615a-47aa-4287-a958-e662e7f681fe"
   {:compute-ref #uuid "e2db7c53-9940-4d9d-986c-779af6f1d8e1",
    :exec-duration 16549,
    :id "78d7615a-47aa-4287-a958-e662e7f681fe",
    :kind "code",
    :output-log-lines {:stdout 296},
    :runtime [:runtime "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"]},
   "8697ac42-8bc8-4586-afaf-a41829beb77c"
   {:compute-ref #uuid "da6e20fa-e546-4c8b-a218-414ed7e4cc17",
    :exec-duration 9158,
    :id "8697ac42-8bc8-4586-afaf-a41829beb77c",
    :kind "code",
    :output-log-lines {:stdout 3},
    :runtime [:runtime "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"]},
   "8c1ea37e-ca3e-4f9e-b718-cb31d20e8d55"
   {:compute-ref #uuid "48fc9194-1c55-43a2-a85c-513319d2e73b",
    :exec-duration 50258,
    :id "8c1ea37e-ca3e-4f9e-b718-cb31d20e8d55",
    :kind "code",
    :locked? false,
    :output-log-lines {:stdout 66},
    :runtime [:runtime "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"]},
   "933c43f5-42db-4187-81de-b96ec6ce3a86"
   {:compute-ref #uuid "05406a08-27b8-4c33-b62e-ba9b94ac6b3d",
    :exec-duration 170863,
    :id "933c43f5-42db-4187-81de-b96ec6ce3a86",
    :kind "code",
    :locked? false,
    :output-log-lines {:stdout 40514},
    :runtime [:runtime "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"]},
   "99d957af-fae0-41ea-afab-949696fd3c60"
   {:id "99d957af-fae0-41ea-afab-949696fd3c60", :kind "file"},
   "a0c174eb-4650-429b-ad83-d292728d56ac"
   {:id "a0c174eb-4650-429b-ad83-d292728d56ac",
    :kind "reference",
    :link
    [:output "150585be-fcca-4821-a1da-b91a650f258a" "val2014.zip"]},
   "a2a64983-107b-43ec-9a66-79e88e1b533b"
   {:compute-ref #uuid "dfff63a8-6ac7-4f6d-88f0-2932e6ea7c8b",
    :exec-duration 1833,
    :id "a2a64983-107b-43ec-9a66-79e88e1b533b",
    :kind "code",
    :output-log-lines {:stdout 8},
    :runtime [:runtime "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"]},
   "aaaed38c-05c0-4ad8-9d5a-f1e42eb52dd8"
   {:compute-ref #uuid "11f7d8b2-b4dd-4ae8-b9af-59bcfbd82105",
    :exec-duration 1112,
    :id "aaaed38c-05c0-4ad8-9d5a-f1e42eb52dd8",
    :kind "code",
    :output-log-lines {:stdout 6},
    :runtime [:runtime "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"]},
   "ad5390a5-61d9-4750-9c4d-1f4b22449c91"
   {:id "ad5390a5-61d9-4750-9c4d-1f4b22449c91",
    :kind "reference",
    :link
    [:output
     "150585be-fcca-4821-a1da-b91a650f258a"
     "annotations_trainval2014.zip"]},
   "b531084f-b926-4cba-b7a4-3f5e92438b29"
   {:compute-ref #uuid "767af395-f880-4e1f-a774-277744eda155",
    :exec-duration 8113,
    :id "b531084f-b926-4cba-b7a4-3f5e92438b29",
    :kind "code",
    :output-log-lines {:stdout 32},
    :runtime [:runtime "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"]},
   "c52b7b53-f891-40b3-b443-eba915e57b98"
   {:id "c52b7b53-f891-40b3-b443-eba915e57b98",
    :kind "reference",
    :link [:output "71ddd10c-207a-4473-a709-8e43c3eb46d2" nil]},
   "d3508725-8cdf-416b-9ca3-8f97cca1a142"
   {:id "d3508725-8cdf-416b-9ca3-8f97cca1a142",
    :kind "reference",
    :link
    [:output "150585be-fcca-4821-a1da-b91a650f258a" "test2014.zip"]},
   "d9732e58-bbf7-40c8-81e0-60e79db38101"
   {:id "d9732e58-bbf7-40c8-81e0-60e79db38101",
    :kind "github-repository",
    :ref "master"},
   "f14263af-258c-4c75-bfa9-d8334becc3a0"
   {:compute-ref #uuid "2fa5b049-72d3-476f-ab0c-cc06f3ab3a36",
    :exec-duration 455,
    :id "f14263af-258c-4c75-bfa9-d8334becc3a0",
    :kind "code",
    :output-log-lines {:stdout 330},
    :runtime [:runtime "3fa4d222-d6e3-4a8f-9cd1-ac490a7b3325"]}},
  :nextjournal/id #uuid "02df7717-0b3f-47d8-9a00-c0b5e372c244",
  :article/change
  {:nextjournal/id #uuid "5ecd07f6-e3b4-4e4b-91a6-ff914b7dcbad"}}}

```
</details>
