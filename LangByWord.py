# source language: python 3.5.1
import os
import math
import pickle


class LangByWord:
    """ A class for training and testing a language identification model.
    The class can train a model from data in a training directory, build a model and then tested using the
    standard test file format. The format of the training directory is:
    1. One file per language
    2. Each file name is of the form lang-xx.txt where xx is on of 21 language codes ('bg','cs','da',... )
    3. All files in one directory

    The sequence of use is:
    1. construct the object: my_obj = LangByWord()
    2. train from files in the training directory: my_obj.train(my_training_dir)
    3. test on the test file: my_obj.test_on_test(my_test_file)

    The test file format is the standard one supplied by the problem authors and is of the form:
        <language code>\t<language text>
        where:
            <language code> is a two character language indentifier
            <language text> is a unicode string from the language

    In addition the model can be saved as a file (save_object_to_file) and loaded from a file (load_object_from_file)
    """

    def __init__(self):
        """ Constructor for LangByWord
        """
        self.training_complete = False  # used to prevent testing before training is complete
        self.init_data_dir = ''  # the name of the training directory is saved here
        self.lang_word_count = {}
        # word counts for each language and each word of the language
        self.lang_word_prob = {}  # word probabilities for each language and each word of the language
        self.lang_sentence_count = {}  #: :type dict of str, dict of str, int
        # the number of sentences processed in each language
        self.lang_total_word_count = {}  # the number of words processed in each language
        self.out_of_vocab_prob = 1e-10  # used to set the probability of words not found in a language

    @staticmethod
    def sentence_to_word_list(sentence: str):
        """ Return a list of the words in the sentence. Strip punctuation from the beginning and end of each word.
        :param sentence: a string representation of a sentence of a language
        :return: a list of the words in the sentence
        """
        sentence = sentence.lower()  # convert to lower case
        sentence = sentence.rstrip('\n')  # remove the end of line from the sentence
        words = sentence.split(' ')  # split the sentence into words using a blank space as the word separator
        new_words = [word.strip('.,()[]-!:?;\\"') for word in words]  # remove punctuation
        return new_words

    def train(self, training_data_dir: str, max_words_per_lang=0, report_freq=0):
        """ Train the language identification model. The language identification model is a dictionary of probabilities
        for each word. There is one dictionary per language. The training data directory contains a file for each
        language. This corpus is used to first count occurrences of words in the language and then compute probabilities
        of words from the counts.
        :param training_data_dir: A directory of training files. One file per language.
        :param max_words_per_lang: The maximum number of words from a language to use for training. A value of 0 will
            cause all of the data to be used. This can be used to limit the training data for speeding up development
            testing time.
        :param training_data_dir: the name of the training data directory
        :param max_words_per_lang: used to limit the amount of training data used, 0 means no limit
        :param report_freq: used to control the reporting output during training,
            0 means no reporting.
        :return: returns nothing.
        """
        # Check for the existence of the training directory
        if not os.path.isdir(training_data_dir):
            print('training directory does not exists:', training_data_dir)
            return

        # save parameters for other function calls
        self.init_data_dir = training_data_dir

        # The training directory has a file for each language. Each file contains the corpus of the language.
        # Process each directory building a word count dictionary for each language.
        for ix, file in enumerate(os.listdir(training_data_dir)):
            lang_name = file.split('-')[1].split('.')[0]  # get the language's name (string representation)
            self.lang_total_word_count[lang_name] = {}  # counts of the words of each language
            full_file_name = os.path.join(training_data_dir, file)
            fh = open(full_file_name, 'r')
            self.lang_sentence_count[lang_name] = 0
            self.lang_total_word_count[lang_name] = 0
            self.lang_word_count[lang_name] = {}
            for sentence in fh:
                self.lang_sentence_count[lang_name] += 1
                # Convert the sentence to a list of words and count each word's occurrence in the language
                words = self.sentence_to_word_list(sentence)  # create a list of words in the sentence
                for word in words:
                    self.lang_total_word_count[lang_name] += 1
                    if word in self.lang_word_count[lang_name]:
                        self.lang_word_count[lang_name][word] += 1  # increment the number occurrences in this language
                    else:
                        self.lang_word_count[lang_name][word] = 1  # initial occurrence of this word in the language
                # Check for early training termination on this language
                if max_words_per_lang > 0:
                    if len(self.lang_word_count[lang_name].keys()) > max_words_per_lang:
                        break
                if report_freq > 0 and self.lang_sentence_count[lang_name] % report_freq == 0:
                    print('language:', lang_name,
                          'words processed:', self.lang_total_word_count[lang_name],
                          'sentences processed:', self.lang_sentence_count[lang_name])
            print('final stats:', lang_name, 'unique words:', len(self.lang_word_count[lang_name].keys()),
                  'sentences:', self.lang_sentence_count[lang_name])

        # Compute the probabilities of each word of a language by dividing the total number of counts of the word
        # in the language's corpus by the total number of words in the corpus.
        for lang in self.lang_word_count.keys():
            self.lang_word_prob[lang] = {}  # word probabilities for each language
            sum_counts = 0
            for word in self.lang_word_count[lang]:
                sum_counts += self.lang_word_count[lang][word]
            for word in self.lang_word_count[lang]:
                self.lang_word_prob[lang][word] = self.lang_word_count[lang][word] / sum_counts

        # Mark the object as having completed training
        self.training_complete = True

        # Compute the out of vocabulary probability by finding the minimum word probability in all languages
        # and reducing it by a factor. This is used during testing to assign a probability to a word that is not
        # found in the vocabulary (and hence cannot be estimated).
        self.out_of_vocab_prob = self.find_minimum_word_prob()

    def sentence_log_prob(self, sentence: str) -> (str, float):
        """ Find the language which maximizes the sentence probability.
        :param sentence: a string representing a sentence
        :return: the language and the probability of the sentence
        """
        words = self.sentence_to_word_list(sentence)  # create a list of the words in the sentence
        # Create a list of language name, log prob pairs
        pl = [(l, self.sentence_log_prob_from_language(words, l)) for l in self.lang_word_prob.keys()]
        probs = [p[1] for p in pl]  # create a list of log probabilities only
        max_prob = max(probs)  # find the maximum log probability
        max_i = probs.index(max_prob)  # get the index of the maximum
        return pl[max_i]  # return the most likely language and the sentence probability given it is from this language

    def sentence_log_prob_from_language(self, word_list: [str], lang: str):
        """ Compute the log probability of a list of words from a given language 'lang'. The log probabilities are
        summed rather than the probabilities multiplied in order to prevent underflow.
        :param word_list: input word list
        :param lang: language to compute probability relative to
        :return: the log probability of the word list relative to the specified language
        """
        log_prob_sum = 0.0
        for word in word_list:
            if word in self.lang_word_prob[lang]:
                log_prob = math.log(self.lang_word_prob[lang][word])
            else:
                log_prob = math.log(self.out_of_vocab_prob)
            log_prob_sum += log_prob
        return log_prob_sum

    def test_on_train(self):
        """ Test the training on itself. This is used for development test and debug only.
        """
        error_count = 0
        sentence_count = 0
        for ix, file in enumerate(os.listdir(self.init_data_dir)):
            lang_name = file.split('-')[1].split('.')[0]  # get the language's name (string representation)
            full_file_name = self.init_data_dir + '/' + file
            fh = open(full_file_name, 'r')
            line_count = 0
            for line in fh:
                lang_match = self.sentence_log_prob(line)
                if lang_match != lang_name:
                    error_count += 1
                sentence_count += 1
                line_count += 1
                if line_count >= self.lang_sentence_count[lang_name]:
                    break
                if line_count % 10000 == 0:
                    print('sentence count=', sentence_count, 'error_count=', error_count, 'error rate=',
                          100 * error_count / sentence_count)
        print('sentence count=', sentence_count, 'error_count=', error_count, 'error rate=',
              100*error_count/sentence_count)

    def test_on_test(self, test_file: str, report_freq=0):
        """ After training is complete this function can be called to compute the error rate of the test data.
        The test_file must be in the format of a single sentence per line ending in newline (\n). The line's language
        is given by the first two characters of the line. A tab (\t) is used to separate the language id from the
        languages text.
        :parm test_file: the name of the test file 
        :parm report_freq: the number of sentences per status output, 0 means no status output
        """
        if not self.training_complete:
            print('***error***, training must be done before the function can be called.')
            return

        if not os.path.isfile(test_file):
            print('***error***, the test file does not exist:', test_file)
            return

        # Open the test file and process each line (sentence) of the file
        fh = open(test_file, 'r')
        error_count = 0  # total errors
        sentence_count = 0  # total sentences
        for line in fh:
            lang_id, lang_text = line.split('\t', 2)  # split into two parts using the tab
            prob_lang, prob = self.sentence_log_prob(lang_text)
            if prob_lang != lang_id:
                error_count += 1
                print(lang_id + '->' + prob_lang + '  \"' + lang_text.rstrip('\n') + '\"')
            sentence_count += 1
            if report_freq > 0 and sentence_count % report_freq == 0:
                print(error_count, 'errors in', sentence_count, 'sentences.',
                      'error percent:', '{0:.4f}'.format(100*error_count/sentence_count))
        fh.close()
        print('Error count:', error_count, 'sentence count:', sentence_count,
              'percent error rate: ', '{0:.4f}'.format(100*error_count/sentence_count))

    def find_minimum_word_prob(self):
        """ Find the minimum word probability over all vocabularies. This function can be used to compute the
        out_of_vocab_prob from the training data.
        :return: the minimum probability of all words.
        """
        if not self.training_complete:
            print('Training is not complete. find_minimum_word_prob_stopping.')
            return self.out_of_vocab_prob

        min_probs = [min(self.lang_word_prob[lang].values()) for lang in self.lang_word_prob.keys()]
        return min(min_probs)

    def print_most_prob_words(self):
        """ Print the most probable word of each language. This is useful for testing the 'train' function.
        """
        if not self.training_complete:
            print('Training is not complete. print_most_prob_words stopping.')
            return

        print('Most probable word of each language:')
        for lang in self.lang_word_count.keys():
            max_prob = 0.0
            max_prob_word = 'None???'
            for word in self.lang_word_prob[lang]:
                if self.lang_word_prob[lang][word] > max_prob:
                    max_prob = self.lang_word_prob[lang][word]
                    max_prob_word = word
            print('language:', lang, 'word:', max_prob_word, 'prob:', '{0:.4f}'.format(max_prob))

    def save_object_to_file(self, file_name: str):
        """ Save this object to a file
        """
        with open(file_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_object_from_file(file_name: str):
        """ Construct an object of type LangByWord from a file
        """
        with open(file_name, 'rb') as input_file:
            obj = pickle.load(input_file)
            return obj

    @staticmethod
    def self_test():
        """ Test case for this module
        """
        print('Begin self_test.')
        lo = LangByWord()
        lo.train('/Users/frank/data/LanguageDetectionModel/exp_data', 10000)
        lo.print_most_prob_words()
        object_file = 'LbyW_obj.pck'
        lo.save_object_to_file(object_file)
        lo2 = LangByWord.load_object_from_file(object_file)
        print('Minimum word probability:', lo2.find_minimum_word_prob())
        lo2.test_on_test('/Users/frank/data/LanguageDetectionModel/europarl.test')
