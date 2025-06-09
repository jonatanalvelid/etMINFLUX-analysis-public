import specpy
import os
import time

# REQUIRES A WORKING AND OPEN IMSPECTOR APPLICATION WHEN RUNNING. Converts all MINFLUX datasets in .msr files in a give folder to .npy files

def open_file(imspector, data_path, basefolders, sleeptime=1):
    imspector.open(data_path)
    # check for new datafolder that has appeared - this is the opened dataset, and that way we can get the tag of the dataset
    data_tags = list(set(os.listdir('C:\\Data'))-set(basefolders))
    time.sleep(sleeptime)
    return data_tags

def save_npy(imspector, data_tags, folder,sleeptime=1):
    for data_tag in data_tags:
        # dataset tag (not visible name) needed to set the correct data source in the minflux data panel
        imspector.value_at('minflux_data_panel/data/id', specpy.ValueTree.Status).set(data_tag)
        imspector.value_at('minflux_data_panel/export/file', specpy.ValueTree.Status).set(folder)
        imspector.value_at('minflux_data_panel/export/as_npy', specpy.ValueTree.Status).trigger()
        time.sleep(sleeptime)

def close_meas(imspector,sleeptime=1):
    meas = imspector.active_measurement()
    imspector.close(meas)
    time.sleep(sleeptime)

### USER PARAMS ###   
datafolder = os.path.join('C:\\Users\\UserName\\Documents\\Data\\MINFLUX\\DataFolder\\')
###################

imspector = specpy.get_application()
data_basefolders = os.listdir('C:\\Data')  # base datafolders in Imspector
savefolder = datafolder
sleeptime = 0.2

datasets = os.listdir(datafolder)
datasets = [os.path.join(datafolder,path) for path in datasets if path.endswith('.msr')]

for filepath in datasets:
    datatags = open_file(imspector, filepath, data_basefolders, sleeptime=sleeptime)
    save_npy(imspector, datatags, savefolder, sleeptime=sleeptime)
    close_meas(imspector, sleeptime=sleeptime)
