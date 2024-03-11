
# Fake-News-Detection-using-Mixtral-7b


Mistral 7B, a powerful language model, can help in the fight against fake news. It analyzes news articles, searching for inconsistencies, unreliable information, and biased language that often characterize fake news.

As we're dealing with LLM, even after quantization it wouldn't be possible to run on consumer hardware. Hence I have used 1x RTX A4500 on Runpod. 

I incorporated FAISS database to store various news articles. Using RAG(Retrieval Augmented Generation) news articles similar to the fake news are retrieved and then both are sent as a prompt into mixtral pipeline.

The code is as belows:


## Code

Firstly install all the dependencies using pip, don't worry about what they do for now. Later in the code they're explained.

```bash
!pip install -q -U torch transformers langchain playwright html2text sentence_transformers faiss-cpu scipy
!playwright install -deps
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 trl==0.4.7
```

Here we are importing the various libraries from langchain which help us to perform web scrapping and extracting relevant text from html code.
```bash
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
import nest_asyncio
nest_asyncio.apply()
```
We then import various classes from the state-of-the-art transformers library.

```bash
import torch
import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
```
We mention our model name, and then load the respective tokenizer.
```bash
model_name="mistralai/Mistral-7B-Instruct-v0.1"
tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"
```
This checks if we have CUDA supported GPUs and prints the available. I'm using RTX A4500 which supports CUDA.
```bash
if torch.cuda.is_available():
    num_devices=torch.cuda.device_count()
    for i in range(num_devices):
        print(f"CUDA Device{i}:{torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available")

```
The code uses BitsAndBytesConfig to configure the model for quantization. It mentions to load the weights in 4bit format, using "nf4" quantization method, and uses "float16" for computations. It then also checks if GPU supports "float16". For more details about this visit huggingface quantization documentation.
```bash
compute_dtype=getattr(torch, "float16")
bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
major, _=torch.cuda.get_device_capability()
if major>=8:
    print("Yes it supports")
```
Now the quantized model is loaded, this will take some time to run.
```bash
model=AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
```
This code will tell us how many parameters are there in the model. It is in billions.
```bash
def no_parameters(model):
    all_model_params=0
    for _, param in model.named_parameters():
        all_model_params+=param.numel()
    return f"total number of parameters: {all_model_params}"

res=no_parameters(model)
print(f"{res}")
```
Now using the langchain's HuggingFacePipeline we create a pipeline which  makes it easy for inferences(in other words, to send the queries to the model).
```bash
text_generation_pipeline=pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    return_full_text=True,
    max_new_tokens=1000,
)
mistral_llm=HuggingFacePipeline(pipeline=text_generation_pipeline)
```
Don't get scared by looking at the code. We are basically passing the url of trusted sources of news, which then brings all the HTML code of the web page using AsyncChromiumLoader. Then we transform the HTML code into relevant text (text containing the news) using Html2TextTransformer. Which is further split into chunks of documents.
```bash
article= ["https://timesofindia.indiatimes.com/india/timestopten.cms"]
loader=AsyncChromiumLoader(article)
docs=loader.load()
html2text=Html2TextTransformer()
docs_transformed=html2text.transform_documents(docs)
text_splitter=CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunked_documents=text_splitter.split_documents(docs_transformed)
```
Now we're storing the documents in FAISS and we're embedding the sentences in the documents into vectors using HuggingFaceEmbeddings. Then we use FAISS retriever to retrieve the vectors similar to the fake news provided by us.

```bash
db=FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
retriever=db.as_retriever()
```
Now we create a prompt template as they provide a structured way to transform simple text inputs into a format that's optimized for our use case. They guide the model to generate more relevant and consistent outputs.

```bash
prompt_template="""
### [INST] Instruction: Check whether the given news is correct or not by referencing the original news given below:

{original_news}

### News
{News} [/INST]
"""
prompt=PromptTemplate(
    input_variables=["original_news", "News"], template=prompt_template,)
```
We create a rag_chain where the retriever finds relevant information which is then processed by an llm_chain which sends the retrieved information along with user prompt into the mistral_llm.
```bash
llm_chain=LLMChain(llm=mistral_llm, prompt=prompt)
rag_chain=({"original_news":retriever, "News":RunnablePassthrough()} | llm_chain)
```
Finally, we use rag_chain.invoke to send the user prompt or the fake news, which is then sent to llm_chain and it outputs the result. 
```bash
result = rag_chain.invoke("Unfortunately Oppenheimer film couldn't get any awards at the Oscars this year")
output = f"News: {result['News']}\nText: {result['text']}"
print(output)
```
I prompted that Oppenheimer movie didn't win any Oscars, but it detected that it is fake News. Anyways Have you watched Oppenheimer?
## Appendix

Hugging Face Transformers: https://huggingface.co/transformers/

Quantization: https://medium.com/@abonia/llm-series-quantization-overview-1b37c560946b

FAISS: https://faiss.ai/

LangChain: https://www.langchain.com/


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/palthyadeepmalik)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/Deepmalik177)

