import os
import csv


"""
Script used to reorganize features into smaller files to avoid Out Of Memory problems.

    This script will reorganize features into:
        v_<<ClassName>>_<<VideoId>>_<<VideoShard>>
            e.g. v_ApplyEyeMakeup_g01_c01.txt
        
    Each video will contain all frames extracted from extract.py script in a csv format.
"""

if __name__ == '__main__':

    filename = 'features_file.txt'

    current_path         = os.path.dirname(__file__)
    feature_file_path    = os.path.join(current_path, 'data')

    save_path            = 'data/'

    features_file_complete_path = os.path.join(feature_file_path, filename)

    with open(features_file_complete_path, 'r') as file:
        pool_csv = csv.reader(file)

        for idx, line in enumerate(pool_csv):
            videoName = line[0].rsplit('_', 1)[0]

            saveFile = save_path + videoName + '.txt'
            outputFile_openMode = 'a' if os.path.isfile(saveFile) else 'w'

            with open(saveFile, outputFile_openMode) as outputFile:
                features = ','.join(map(str, line))
                outputFile.write('%s\n' % features)


        