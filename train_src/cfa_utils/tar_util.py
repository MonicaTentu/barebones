import tarfile
from os import listdir, unlink
from os.path import isfile, join, isdir, splitext
import shutil

def extract_tarball(tarball, extract_path):
    tb = tarfile.open(tarball)
    tb.extractall(path=extract_path)

def extract_tarball_directory(tarball_dir, extract_path, file_ext, final_path):
    # get the list of tarballs
    # - extract each of them
    tarball_list = []
    tarball_count = 0
    for tb in listdir(tarball_dir):
        tb_fullpath = join(tarball_dir, tb)
        if isfile(tb_fullpath) and tarfile.is_tarfile(tb_fullpath):
            extract_tarball(tb_fullpath, extract_path)
        tarball_count = tarball_count + 1

    # ASSUMES (dangerously!) that there is only one subdirectory level
    # get all subdirectories
    subdir_list = []
    subdir_count = 0
    for sd in listdir(extract_path):
        sd_fullpath = join(extract_path, sd)
        if isdir(sd_fullpath):
            subdir_list.append(sd_fullpath)
            subdir_count = subdir_count + 1

    # include the top directory - in case there was not subdirectory
    subdir_list.append(extract_path)

    # now move all extracted globs to the final_path
    #  - these are full subdirectory paths in this list
    for sd in subdir_list:
        for f in listdir(sd):
            file_fullpath = join(sd, f)
            if isfile(file_fullpath):
                filename, file_extension = splitext(file_fullpath)
                if file_extension.lower() == file_ext:
                    shutil.move(file_fullpath, final_path)

    # delete anything in the extract_path - this was just temporary
    for the_file in listdir(extract_path):
        file_path = join(extract_path, the_file)
        try:
            if isfile(file_path):
                unlink(file_path)
            elif isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    return subdir_count, tarball_count
