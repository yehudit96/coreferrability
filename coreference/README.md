In this project there are scripts in order to prepere tweets pair corpus for coreference resolution system format.
The coref system requires the following:
1. txt file of the entire corpus in the format: (empy line as a documents seperator)
    DOC_ID  SENT_NUM  TOKEN_NUM  TOKEN  COREF_CHAIN*
    
    *If the token isn't a part of coref chain - '-'
2. Entities gold mentions and events gold mentions json files, each mention in the format:
    {"coref_chain": "hit+strike_0_0", 
     "doc_id": "842711424807899137_0hit+strike", 
     "is_continuous": true, #not relevant
     "is_singleton": false, #not relevant
     "mention_type": "HUM", #for event - ACT
     "score": -1.0, #not relevant
     "sent_id": 1, 
     "tokens_number": [0, 1], 
     "tokens_str": "Israeli warplanes"}
3. SRL txt file for the documents.
4. Inner-coref file.
5. Topics list


The data we starts with is 2 directories (good rules and bad rules) with json files containing the tweets pairs per each rule.

The scripts that were written in order to prepere the data:

a. For files 1 and 2, the there are two scripts:
    1. create_coref_pairs: reads all the json rule files from the directories and creates one file with all the data. (The data format detaailed in the script)
       The data is saved to a pickle file OUT_PATH
    2. tweet text.py --input_file=(the pickled file from create_coref_pairs) --text_file=(the path to save the entire corpus) --mentions_dir=(the path to the directopy to save the mantions)
       the script create from the pickled file the entire corpus txt file and the json entities and events mantions
b. For file 3, insted of a script we use the project wd-plus-srl-extraction (TODO: make a script calling to the project)
   The script in the project responsable for the SRL extraction from the data:
   srl_allen.py --corpus_path=(the pickled file from create_coref_pairs) --output_file=(the path for the output file) --data_loader=tweets
c. For file 4, there are also 2 script. The stanford inner-coref system recieves xml file for each document and creates one file with all the inner-coref chained for all the documents.
   1. xml_convert.py --input_file=(the pickled file from create_coref_pairs) --output_dir=(the directory where all the xml files will be saved, sub directory for each rule)
   2. inner_coreference.py --xml_dir=(the output directory from xml_convert) --output_path=(the directopy where the inner coref file will be saved, a file for each rule)
      *******NOTICE - after the inneer-coref files are generated, need to merge all of the together!
d. For file 5, there is one script that recieves the corpus text file and generate a topics list file.
   topics_list.py --corpus_text=corpus text file --topics_path=the output topics list pickled file