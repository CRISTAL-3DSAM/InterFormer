import os
import zipfile
import urllib.request


# ----------------------------------------------------------------------------------------------------------------------
def download_and_unzup(url, root):
    print("Downloading '{}'...".format(url))
    filename = url.split('/')[-1]
    filepath = os.path.join(root, filename)
    urllib.request.urlretrieve(url, filepath)
    print("\tExtracting '{}'...".format(filepath))
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(root)


# ----------------------------------------------------------------------------------------------------------------------
def download_sbu(root):
    """
    Downloads the SBU Kinect Interactions dataset into the specified directory
    Dataset homepage: https://www3.cs.stonybrook.edu/~kyun/research/kinect_interaction/index.html

    """
    sbu_files = [
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s02.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s03.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s07.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s01.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s03.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s06.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s07.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s02.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s04.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s05.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s06.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s02.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s03.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s06.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s02.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s03.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s02.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s03.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s04.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s01.zip',
        'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s03.zip',
    ]

    target_dir = os.path.join(root)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    print("")
    for url in sbu_files:
        download_and_unzup(url, target_dir)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    download_sbu('./SBU-Kinect-Interaction')
