from jsgf import PublicRule, Literal, Grammar
import os
import sys
import codecs



import re

def flatten(l): return [item for sublist in l for item in sublist]

class GenerateJSpeechGrammar(object):
    def __init__(self,phonemes,filename,data_dir):

        # self.sentence=sentence
        self.phonemes=phonemes
        self.data_dir=data_dir
        self.filename=filename
        self.grammar_file=os.path.join(self.data_dir,"{}.jspf".format(self.filename))


    def forcing(self):
        phone_grammar_file=os.path.join(self.data_dir,"{}-forcing.jspf".format(self.filename))
        
        utterance=' SIL '+' [ SIL ] '.join(["{}".format(phone) for phone in self.phonemes])+ ' [ SIL ]'
        self.generate_jsgf(utterance,grammar_label='forcing',rule_label="phonemes", alternative_rules=[],grFn=phone_grammar_file)


    def neighbors(self,language='en'):
        neighbors_grammar_file=os.path.join(self.data_dir,"{}-neighbors.jsgf".format(self.filename))
        phoneset=flatten([word_phone.split() for word_phone in self.phonemes])
        
        utterance=' SIL '+' '.join(["<{}>".format(phone) for phone in phoneset])+ ' [ SIL ]'
        alternative_rules=[]

            
        if language=="en":
            #cmu neiborhood
            for phone in phoneset:
                if phone in cmu_neighbors_dict.keys():
                    alternative_rules.append("<{}> = {} ;".format(phone," | ".join([ alter_phone for alter_phone in cmu_neighbors_dict[phone]])))
        elif language == "fr":
            #liaphone
            for phone in phoneset:
                if phone in liaphon_neighbors_dict.keys():
                    alternative_rules.append("<{}> = {} ;".format(phone," | ".join([ alter_phone for alter_phone in liaphon_neighbors_dict[phone]])))
            
        elif language =="es":
            #spanish simplified
            for phone in phoneset:
                if phone in spanich_neighbors_dict.keys():
                    alternative_rules.append("<{}> = {} ;".format(phone," | ".join([ alter_phone for alter_phone in spanich_neighbors_dict[phone]])))
        elif language=="el":
                for phone in phoneset:
                    if phone in greek_neighbors_dict.keys():
                        alternative_rules.append("<{}> = {} ;".format(phone," | ".join([ alter_phone for alter_phone in greek_neighbors_dict[phone]])))

        else:
            print("error this language {} is not available ".format(language))

        self.generate_jsgf(utterance,grammar_label='neighbors',rule_label="neighbors", alternative_rules=alternative_rules,grFn=neighbors_grammar_file)



    def word(self):
        utterance=' SIL '+' [ SIL ] '.join([ "{}".format(word) for word in self.sentence])+ ' [ SIL ]'
        word_grammar_file=os.path.join(self.data_dir,"{}-word.jspf".format(self.filename))

        self.generate_jsgf(utterance,grammar_label='neighbors',rule_label="neighbors", alternative_rules=[],grFn=word_grammar_file)





    def generate_jsgf(self,literal,grammar_label,rule_label,alternative_rules,grFn):
    	# Create a public rule with the name 'hello' and a Literal expansion 'hello world'.
        
        
        
        rule = PublicRule(rule_label, Literal("{}".format(literal)))

        # Create a grammar and add the new rule to it.
        grammar = Grammar(grammar_label)
        grammar.add_rule(rule)

        with codecs.open(grFn,"w") as grammarFile:
            grammarFile.write(grammar.compile())
            if len(alternative_rules)>0:
                for alternative_rule in alternative_rules:
                    grammarFile.write("{}\n".format(alternative_rule))


    def get_grammar_file_path(self):
        return self.grammar_file
    def set_grammar_file_path(self,new_path):
        self.grammar_file=new_path

    def utterance_grammar_generation(self,language="en"):
        #Grammar utterance
        grammar_file=self.grammar_file
        grammar = Grammar('utt')

        # Phonemes
        phonemes=' SIL '+' [ SIL ] '.join(["{}".format(phone) for phone in self.phonemes])+ ' [ SIL ]'
        rule_phones = PublicRule('phonemes', Literal("{}".format(phonemes)))
        grammar.add_rule(rule_phones)



        with codecs.open(grammar_file,"w") as grammarFile:
            grammarFile.write(grammar.compile())