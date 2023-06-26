import requests
from collections import Counter
from docx import Document
import re
import ast
import pandas as pd
from word2number import w2n
from hugchat import hugchat




class DocumentAnalyzer_content:
    def __init__(self, file_path, max_sentence_length=400, no_of_selected_chunks=4):
        self.file_path = file_path
        self.max_sentence_length = max_sentence_length
        self.no_of_selected_chunks = no_of_selected_chunks
        

    def analyze_document(self):
        def read_docx(file_path,max_sentence_length):
            # Implementation of read_docx() function
            chunk_dict = {"chunks":[], "filenames":[]}
            path_file = file_path.split("/")[-1]
            doc = Document(self.file_path)
            paragraphs = [p.text for p in doc.paragraphs if len(p.text)>1]
            # print(paragraphs[0:20])

            # Join paragraphs into a single string
            document = ' '.join(paragraphs)
            sentences = re.split(r'(?<=[.!?])\s+', document)
            chunks = sentences

            final_chunks = []
            final_filenames = []
            new_data = ""
            index = 0
            index1 = -1
            for c_index,sentence in enumerate(chunks[index:index1]):

                if len(sentence.split(" ")) <= max_sentence_length and len((new_data+sentence).split(" "))<= max_sentence_length:
                    new_data = new_data+sentence
                else:
                    final_chunks.append(new_data)
                    new_data = ""
                    index = c_index

            chunk_dict["chunks"].append(final_chunks)
            chunk_dict["filenames"].append(path_file)


            return chunk_dict


        def chat_with_bot(text):
            # Implementation of chat_with_bot() function
            chatbot = hugchat.ChatBot(cookie_path="/home/ubuntu/cookies.json")
            try:
                response = chatbot.chat(text,temperature=0.2,
                                        top_k=95,
                                        max_new_tokens=512,
                                        )
            except:
                response = "NO"
            #print(text)
            # Create a new conversation

            id = chatbot.new_conversation()
            chatbot.change_conversation(id)


            # Get conversation list
            conversation_list = chatbot.get_conversation_list()

            return conversation_list, response

        def process_dataframe(df):
            print("self.file_path",self.file_path)
            print("*"*50)
            
            total_count = df['phonetic_symbol_count'].tolist()
            
            total_count = sum([int(i) for i in total_count])
            print("total_count",total_count)
            
            
            missing_transposition = df['missing_transpositions'].tolist()
            missing_transposition_lowercase_values = [value.lower() for value in missing_transposition]
            print("missing_transposition_lowercase_values",missing_transposition_lowercase_values)

            quotes = df['Single_or_double_quotes'].tolist()
            quotes_lower_case = [value.lower() for value in quotes]


            Series_comma = df['Series_comma'].tolist()
            Series_comma_lowercase_values = [value.lower() for value in Series_comma]


            American_english = df['American_english'].tolist()
            American_english_lowercase_values = [value.lower() for value in American_english]


            rewrit_document = df['rewrit_document'].tolist()
            rewrit_document_lowercase_values = [value.lower() for value in rewrit_document]


            subject_matter_expertise = df['subject_matter_expertise'].tolist()
            subject_matter_expertise_lowercase_values = [value.lower() for value in subject_matter_expertise]

            mutiple_writting_styles = df['mutiple_writting_styles'].tolist()
            mutiple_writting_styles_lowercase_values = [value.lower() for value in mutiple_writting_styles]

            merged_dict = {}
            name_file = self.file_path.split('/')[-1]
            merged_dict.update({'file_path': name_file})
            print("#"*50)
            # Check missing_transpositions column
            
            merged_dict.update({'phonetic_symbol_count': [total_count]})
                
            if 'yes.' in missing_transposition_lowercase_values:
                print("#"*50)
                merged_dict.update({'missing_transpositions': ['inconsistent']})
            else:
                merged_dict.update({'missing_transpositions': ['consistent']})

            if 'double' in quotes_lower_case and 'single' in quotes_lower_case:
                merged_dict.update({'Single_or_double_quotes': ['inconsistent']})
            else:
                merged_dict.update({'Single_or_double_quotes': ['consistent']})

            if 'no' in  Series_comma_lowercase_values:
                merged_dict.update({'Series_comma': ['inconsistent']})
            else:
                merged_dict.update({'Series_comma': ['consistent']})

            if 'british' in American_english_lowercase_values and American_english_lowercase_values:
                merged_dict.update({'American_english': ['inconsistent has combination of british and american style']})
            else:
                merged_dict.update({'American_english': ['consistent']})

            if 'no' in rewrit_document_lowercase_values:
                merged_dict.update({'rewrit_document': ['consistent']})
            else:
                merged_dict.update({'rewrit_document': ['inconsistent']})

            if 'yes' in subject_matter_expertise_lowercase_values:
                merged_dict.update({'subject_matter_expertise': ['require subject matter expertise']})
            else:
                merged_dict.update({'subject_matter_expertise': ['no-require subject matter expertise']})

            if 'yes' in mutiple_writting_styles_lowercase_values:
                merged_dict.update({'mutiple_writting_styles': ['Has multiple writting styles']})
            else:
                merged_dict.update({'mutiple_writting_styles': ['NO multiple writting styles']})

            merged_df = pd.DataFrame.from_dict(merged_dict)

            return merged_df


        def extract_error_count(response):
            # Implementation of extract_error_count() function
            error_count = re.findall(r'\d+', response.split("\n")[0])

            if error_count:
                # Convert the extracted value to an integer
                error_count = int(error_count[0])
            else :
                # If a numeric value is not found, try to convert words to numbers
                try:
                    error_count = response.split("\n")[0]
                except ValueError:
                    error_count = 0
                    
            return error_count

        prompt_beginners = [

       '''Are there any missing_transpositions in the document? give me response only in  "YES" or "NO":''',
        '''Analyse  the following text  whether it has Single_or_double_quotes and give me respose in "Single" or "double" only?:''',
        '''Analyse  the following text  whether it has Series_comma and give me respose in "Yes" or "No" only?:''',
        '''Analyse  the following text  whether it is in "American_english style" or "British english style and give me respose in "American" or "British" only?:"''',
        '''Analyse the following text do we need to rewrit_document ? give me response only in  "YES" or "NO":''',
        '''count the number of phonetics and symbols present in the document and provide only the count just give me a number'''

        
        ]

        data_dict = read_docx(self.file_path, self.max_sentence_length)

        missing_transpositions = ["NULL"]*self.no_of_selected_chunks
        Single_or_double_quotes = ["NULL"]*self.no_of_selected_chunks
        Series_comma = ["NULL"]*self.no_of_selected_chunks
        American_english= ["NULL"]*self.no_of_selected_chunks
        rewrit_document = ["NULL"]*self.no_of_selected_chunks
        subject_matter_expertise = ["NULL"]*self.no_of_selected_chunks
        mutiple_writting_styles = ["NULL"]*self.no_of_selected_chunks
        phonetic_symbol_count = [0]*self.no_of_selected_chunks
        modified_chunks = []
        for prompt in prompt_beginners:
            for chunk in data_dict["chunks"][0][:self.no_of_selected_chunks]:
                input_text = f"{prompt} {chunk}"
                modified_chunks.append(input_text)
            
#         print(modified_chunks)

        df_data = {
                    'phonetic_symbol_count': phonetic_symbol_count,
                   'missing_transpositions':missing_transpositions,
                   'Single_or_double_quotes':Single_or_double_quotes, 
                   'Series_comma':Series_comma, 
                   'American_english':American_english,
                   'rewrit_document':rewrit_document,
                   'subject_matter_expertise':subject_matter_expertise,
                   'mutiple_writting_styles':mutiple_writting_styles
        }
        
        for index, f_chunks in enumerate(modified_chunks[0:4]):
            # print(f_chunks)
            conversation_list, response = chat_with_bot(f_chunks)
            # numbers = re.findall(r'\d+', response)
            # total = sum(int(num) for num in numbers)
            # total_count += total
            error_count = extract_error_count(response)
            print("response",response)
            print("error_count",error_count)
            
            if "phonetics" in  f_chunks:
                # column_name = 'phonetic_symbol_count'
                phonetic_symbol_count[index % self.no_of_selected_chunks] = error_count if isinstance(error_count, int) else 0

            if "missing_transpositions" in f_chunks:
                # column_name = 'punctuational error_count'
                missing_transpositions[index % self.no_of_selected_chunks]  = error_count
            if "Single_or_double_quotes" in f_chunks:
                # column_name = 'grammatical error_count'
                Single_or_double_quotes[index % self.no_of_selected_chunks] = error_count
            if "Series_comma " in f_chunks:
                # column_name = 'spelling_error_count'
                Series_comma[index % self.no_of_selected_chunks] = error_count
            if "American_english" in f_chunks:
                # column_name = 'missing_articles_count'
                American_english[index % self.no_of_selected_chunks] = error_count
            if "rewrit_document" in f_chunks:
            # column_name = 'missing_articles_count'
                rewrit_document[index % self.no_of_selected_chunks] = error_count
            if "subject_matter_expertise" in f_chunks:
            # column_name = 'missing_articles_count'
                subject_matter_expertise[index % self.no_of_selected_chunks] = error_count
            if "mutiple_writting_styles" in f_chunks:
            # column_name = 'missing_articles_count'
                mutiple_writting_styles[index % self.no_of_selected_chunks] = error_count

        df = pd.DataFrame(df_data)
        processed_df = process_dataframe(df)

        return processed_df

# Usage example:
analyzer = DocumentAnalyzer_content(file_path="/home/ubuntu/cat_poc/data/Aarons-Renamed-r01/15032-5196-FullBook.docx")
result_df = analyzer.analyze_document()
print(result_df)
