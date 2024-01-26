import os
import yaml
from yaml import Loader, Dumper

def edit_content(doc, source, edit_yml = False, yml_path = False, ini_patern= None, end_patern = None, verbose = True):
    content = doc.get(source,False)
    if content:
        if verbose:
            print(f'   {source} content: ',content)
        
        if edit_yml:
            new_value = doc[source].replace(ini_patern, end_patern)
            doc[source] = f"{new_value}"
            if verbose:
                print(f'   {source} new content: ', new_value)
            with open(yml_path, 'w') as f:
                yaml.dump(doc, f)
                
# root = r"C:\Users\Miguel\ImageFinder\app\mlruns"
# ini_patern = 'file:C:/Users/Miguel/ImageFinder/app'
# end_patern = 'file:///app'
# edit_yml = False

def pull_artifacts_from_mlflow(root_path, ini_patern, end_patern, verbose = True, edit_yml = False):

    for path, subdirs, files in os.walk(root_path):
        for name in files:
            if name == 'meta.yaml':
                if verbose:
                    print(os.path.join(path, name))
                yml_path = os.path.join(path, name)
                with open(yml_path) as f:
                    doc = yaml.load(f,Loader)

                edit_content(doc, 'source', edit_yml, yml_path, ini_patern, end_patern, verbose)
                edit_content(doc, 'artifact_uri', edit_yml, yml_path, ini_patern, end_patern, verbose )
                edit_content(doc, 'artifact_location', edit_yml, yml_path, ini_patern, end_patern, verbose )
    if edit_yml:
        print('------------------------------------------------------------')
        print('---------------artifacts were pulled-----------------------')
        print(f'initial pattern: {ini_patern}')
        print(f'new pattern: {end_patern}')
        print('------------------------------------------------------------')
    else:
        print('----------------------just viz-------------------------')