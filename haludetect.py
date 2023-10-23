import csv
import sys
from transformers import AutoModel, AutoTokenizer,AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from pipelines import pipeline

class ModelTokenizerWrapper:
    def __init__(self, model_name, sent_transform,ner):
        # Initialize the tokenizer and model for qa
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Initialize question generation model
        self.nlp = pipeline("question-generation", model="valhalla/t5-small-qg-prepend", qg_format="prepend")

        self.sent_tran = SentenceTransformer(sent_transform)
        #'bert-base-nli-mean-tokens')

        ## Initialize the tokenizer and model for ner 
        self.tokenizer_ner = AutoTokenizer.from_pretrained(ner)
        self.model_ner  = AutoModelForTokenClassification.from_pretrained(ner)                  


    def get_tokenizer(self):
        return self.tokenizer
        
    def get_model(self):
        return self.model

    def get_qg_model(self):
        return self.nlp

    def get_sentence_transfomer(self):
        return self.sent_tran
    
    def get_ner_model(self):
        return self.model_ner
    
    def get_ner_tokenizer(self):
        return self.tokenizer_ner

def generate_df(text,summary):
    '''
    
    '''
    print(" in generate_df ")
    response = nlp(text)
    df = pd.DataFrame.from_records(response)
    df['text'] = text
    df['summary']= summary
    return(df)

def generate_answer(question, context,  model, tokenizer):
    '''
    
    '''
    print(" in generate_answer ")
    prompt = f"Use the following context to answer questions: \n Context :{context} \n Question: {question}"
    inputs = tokenizer( prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return(response)

def combine_entities(entities):
    '''
    
    '''
    # Initialize variables
    combined_entities = []
    current_entity = None
    first_int= True

    # Iterate through the entities
    for entity in entities:
        #print(entity)
        entity_prefix , entity_type  = entity['entity'].split('-')

        #print(entity_type, entity_prefix)
        if entity_prefix == 'B':
            
            # If it's the beginning of an entity, create a new entity
            if current_entity:
                combined_entities.append(current_entity)
            current_entity = {'entity': entity_type, 'word': entity['word']}
            
        elif entity_prefix == 'I':
            # If it's inside an entity, append the word to the current entity
            if current_entity:
                if(entity['word'].startswith("##")):
                    current_entity['word'] += ''+ entity['word'].replace("##",'')
                else:
                    current_entity['word'] += ' '+ entity['word']
        else:
            # If it's outside an entity, add the current entity to the result and reset it
            if current_entity:
                combined_entities.append(current_entity)
            current_entity = None

    # Add the last entity if it exists
    if current_entity:
        combined_entities.append(current_entity)

    # Print the combined entities
    return(combined_entities)

def get_entity_matches(id,text,summary,ner_nlp):
    '''
    
    '''
    ner_results = ner_nlp(text)
    combined_entities = combine_entities(ner_results)
    df_combined_entities = pd.DataFrame.from_records(combined_entities)

    ner_results_summary= ner_nlp(summary)
    combined_entities_summary = combine_entities(ner_results_summary)
    df_combined_entities_summary = pd.DataFrame.from_records(combined_entities_summary)
    #print(df_combined_entities_summary)
    items_not_in_summary = [item for item in df_combined_entities['word'] if item not in df_combined_entities_summary['word'] ]
    items_not_in_original = [item for item in df_combined_entities_summary['word'] if item not in df_combined_entities['word'] ]
    items_present_in_both = [item for item in df_combined_entities['word'] if item in df_combined_entities_summary['word'] ]

    ind1= len(items_not_in_summary)
    ind2=len(items_not_in_original)
    ind3= len(items_present_in_both)
    
    text_entities=','.join(df_combined_entities['word'] )
    summary_entities = ",".join( df_combined_entities_summary['word'])
    
    return({
        'id':id,
        "text_entities":text_entities,
        "summary_entities":summary_entities,
        "items_not_in_summary_cnt":ind1,
        "items_not_in_summary" : items_not_in_summary,
        "items_not_in_original_cnt" :ind2,
        "items_not_in_original" :items_not_in_original,
        "items_present_in_both_cnt":ind3,
        "items_present_in_both" : items_present_in_both
    })



def process_csv_verification_questions( data, tokenizer, model ,nlp , sent_tran):
    '''
    
    '''
    print(" in process_csv ")
    try:
        
        l = []
        for i, row in data.iterrows():
            
            df = generate_df(row['text'],row['summary'])
            df['response_text']= [' '.join(generate_answer(question, text , model, tokenizer)) for question, text in zip(df['question'],df['text'])]
            df['response_summary']= [' '.join(generate_answer(question, summary ,  model, tokenizer)) for question, summary in zip(df['question'],df['summary'])]
            temp1 = sent_tran.encode(df['response_text'])
            temp2 = sent_tran.encode(df['response_summary'])
            df['cosine_score'] = [cosine_similarity(x.reshape(1, -1),y.reshape(1, -1))[0][0] for x, y in zip(temp1,temp2)]
            df['id'] = row['id']
            l.append(df)
        return(l)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")    

def process_csv_ner(data,model_ner, tokenizer_ner):
    '''
    
    '''
    from transformers import pipeline as p

    l =[]
    ner_nlp = p("ner", model=model_ner, tokenizer=tokenizer_ner)
    for i, row in data.iterrows():
        l.append(get_entity_matches(row['id'],row['text'],row['summary'], ner_nlp))
    return (pd.DataFrame(l))

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python haludetect.py path_to_csv_file")
        sys.exit(1)
    else:
        file_path = sys.argv[1]
        print("Processing..")
    
        model_name = "google/flan-t5-xl"
        sent_transform = "bert-base-nli-mean-tokens"
        model_ner = "dslim/bert-large-NER"
        
        wrapper = ModelTokenizerWrapper(model_name, sent_transform,model_ner)
        
        tokenizer = wrapper.get_tokenizer()
        model = wrapper.get_model()
        
        nlp = wrapper.get_qg_model()
        sent_tran = wrapper.get_sentence_transfomer()

        model_ner = wrapper.get_ner_model()
        tokenizer_ner = wrapper.get_ner_tokenizer()
    
        data = pd.read_csv(file_path)
        data['id'] = range(1,len(data)+1)

        ## Method1
        output_1 = process_csv_verification_questions(data, tokenizer, model ,nlp , sent_tran) 
        df1 = pd.concat(output_1)
        print(df1.columns)

        ##Method2
        df2 = process_csv_ner(data,model_ner, tokenizer_ner)
    
        df1.to_csv("output_method1.csv", index=False)
        df2.to_csv("output_method2.csv", index=False)

      

        
