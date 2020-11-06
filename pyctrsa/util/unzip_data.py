import zipfile

#filename = 'BaeLuck_2018jn_data.zip'
#data_dir = '/Users/zitonglu/Downloads/'
#filepath = data_dir + filename

def unzipfile(filepath, data_dir):

    with zipfile.ZipFile(filepath, 'r') as zip:
        zip.extractall(data_dir)
    print("Unzip completes!")

# test
#unzipfile(filepath, data_dir)