import os


class BuildTrainingDataFiles:
    """ A class for pre-processing the language identification task.
    The function start_building should be run on the initial data-set to:
    1. remove unwanted lines.
    2. create a file for each language to allow for more efficient access during training.
    Note: The base_input_dir should be in the form download from http://www.statmt.org/europarl/.
    The 'txt' directory should contains 21 subdirectories ('bd','cs','da','de',...)
    Each directory contains a set of text files for that language.
    """

    @staticmethod
    def filter_data(txt):
        """ Remove unwanted lines from txt
        :param txt: input text to be processed
        :return: new text with unwanted lines removed
        """
        lc = 0
        new_txt = ''
        # Process each line delimitinated by a newline
        for line in txt.split('\n'):
            if len(line) > 0:
                # print(lc, line)
                lc += 1
                if line[0] == '<':  # ignore lines beginning with "<"
                    continue
                if line[0] == '(':  # ignore lines beginning with "("
                    continue
                new_txt += line + '\n'  # add the line to the output text
        return new_txt

    @staticmethod
    def start_building(base_input_dir, base_output_dir):
        """ Pre-process all the input files to produce a new directory.
        :param base_input_dir: location of files to be processed.
        :param base_output_dir: location of the output files
        :return: nothing.
        """
        # Check that the input directory exists
        if not os.path.isdir(base_input_dir):
            print('***error***, the input directory does not exist:', base_input_dir)
            return

        # Create the output directory if it does not exist
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

        # Process all the files in each top level directory. There should be one directory for each
        # language to be processed.
        for sub_dir in os.listdir(base_input_dir):
            sub_dir_full_name = os.path.join(base_input_dir, sub_dir)  # get the name of the language sub-directory
            print('Processing directory:',sub_dir_full_name)
            language_txt = ''
            # Process each file within the sub-directory
            for file_name in os.listdir(sub_dir_full_name):
                full_file_name = os.path.join(sub_dir_full_name, file_name)
                try:
                    fd = open(full_file_name, 'r')
                    txt = fd.read()
                    fd.close()
                except UnicodeDecodeError:
                    print('Error on Unicode Decode. File will be ignored:', full_file_name)
                else:
                    filt_txt = BuildTrainingDataFiles.filter_data(txt)
                    language_txt += filt_txt
   
            # Write out a file for this language
            language_file_name = os.path.join(base_output_dir, 'lang-' + sub_dir + '.txt')
            fh = open(language_file_name, 'w')
            fh.write(language_txt)
            fh.close()

    @staticmethod
    def self_test():
        base_input_dir = '/home/frank/data/LanguageDetectionModel/txt'
        base_output_dir = '/home/frank/data/LanguageDetectionModel/exp_data_test'
        build_obj = BuildTrainingDataFiles()
        build_obj.start_building(base_input_dir, base_output_dir)
