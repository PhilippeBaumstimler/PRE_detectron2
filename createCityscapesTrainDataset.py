# Création d'une nouvelle dataset d'entrainement cityscapes_pm,
# à partir des scripts de création d'instances et de labels du
# module cityscapesscripts.
# cityscapes_pm instancie et regroupe les données selon leur caractère
# potentiellement mobile ou nécessairement statique.
#
# cityscapes_pm possède la même architecture que cityscapes et regroupe
# les données suivantes
#   a) *labelIds.png          : la classification par ID des objets potentiellement mobiles
#   b) *instanceIds.png       : la classe et l'instance par ID des objets potentiellement mobiles
#   c) *polygons.json       : la segmentation des objets par polygone de l'ensemble des objets de l'image
#   d) *leftImg8bit.png     : l'image originale prise en situation réelle

import os
import glob
import sys
import cv2 as cv
import json

from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.preparation.json2instanceImg import json2instanceImg
from cityscapesscripts.preparation.json2labelImg import json2labelImg

def main():
    # Cityscapes directory path
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    # Récupération des fichiers de ground truth
    searchFine   = os.path.join( cityscapesPath , "gtFine"   , "*" , "*" , "*_gt*_polygons.json" )
    searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" )

    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()
    filesCoarse = glob.glob( searchCoarse )
    filesCoarse.sort()

    # concatenate fine and coarse
    files = filesFine + filesCoarse
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        printError( "Did not find any files. Please consult the README." )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))
    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # create the output filename
        dst_instance = f.replace( "_polygons.json" , "_instanceIds.png" )
        dst_instance = dst_instance.replace("cityscapes", "cityscapes_pm")
        dst_label = f.replace( "_polygons.json" , "_labelIds.png" )
        dst_label = dst_label.replace("cityscapes", "cityscapes_pm")
        dst_poly = f.replace("cityscapes", "cityscapes_pm")
        if not os.path.exists(os.path.dirname(dst_label)):
            os.makedirs(os.path.dirname(dst_label))
        try:
            json2instanceImg( f , dst_instance , "trainIds" )
            json2labelImg( f , dst_label , "trainIds" )
            FILE = open(f)
            tmp = json.load(FILE)
            with open(dst_poly, 'w') as jf:
                json.dump(tmp, jf)
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()
    progress = 0
    # Recopie de leftImg8bit dans la nouvelle base de données
    path_leftImg8bit   = os.path.join( cityscapesPath , "leftImg8bit"   , "*" , "*" , "*leftImg8bit.png" )
    files_leftImg8bit = glob.glob(path_leftImg8bit)
    files_leftImg8bit.sort()
    if not files:
        printError( "Did not find any leftImg8bit files." )
    print("Saving {} leftImg8bit files".format(len(files_leftImg8bit)))
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files_leftImg8bit:
        dst_img = f.replace("cityscapes", "cityscapes_pm")
        if not os.path.exists(os.path.dirname(dst_img)):
            os.makedirs(os.path.dirname(dst_img))
        img = cv.imread(f)
        cv.imwrite(dst_img, img)
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()





if __name__ == "__main__":
    main()
