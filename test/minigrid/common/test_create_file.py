import os

from multiprocessing import Pool

from big_rl.minigrid.common import create_unique_file, create_unique_directory


##################################################
# Pickleable functions for multiprocessing
##################################################


def _create_file(directory):
    create_unique_file(directory, "test", "txt")


def _create_file_non_atomic(directory):
    name = "test"
    extension = "txt"

    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f'{name}.{extension}')
    index = 0
    while True:
        if os.path.exists(filename):
            index += 1
            filename = os.path.join(directory, f'{name}-{index}.{extension}')
            continue
        os.open(filename,  os.O_CREAT)
        return filename


def _create_directory(directory):
    create_unique_directory(directory, "test")


##################################################
# Files
##################################################


def test_create_unique_file(tmpdir):
    directory = tmpdir.mkdir("test")

    # Check that there are no files in the directory
    assert len(directory.listdir()) == 0

    # Create a file
    file = create_unique_file(directory, "test", "txt")
    assert os.path.basename(file) == "test.txt"
    assert os.path.exists(file)

    # Check that the file was created
    assert len(directory.listdir()) == 1


def test_create_unique_file_duplicate_name(tmpdir):
    directory = tmpdir.mkdir("test")

    # Create a file
    file1 = create_unique_file(directory, "test", "txt")
    file2 = create_unique_file(directory, "test", "txt")
    file3 = create_unique_file(directory, "test", "txt")

    assert file1 != file2
    assert file1 != file3
    assert file2 != file3

    assert os.path.basename(file1) == "test.txt"
    assert os.path.basename(file2) == "test-1.txt"
    assert os.path.basename(file3) == "test-2.txt"

    # Check that the file was created
    assert len(directory.listdir()) == 3


def test_create_unique_file_race_condition(tmpdir):
    """ Run `create_unique_file` many times in parallel to make sure different processes don't accidentally try to use the same file name. """
    directory = tmpdir.mkdir("test")

    with Pool(10) as p:
        p.map(_create_file, [directory] * 100)

    # Check that the file was created
    assert len(directory.listdir()) == 100


def test_create_unique_file_race_condition_failure(tmpdir):
    """ Test with a non-atomic file creation function to make sure that the test fails if the file creation is not atomic. """
    directory = tmpdir.mkdir("test")

    with Pool(10) as p:
        p.map(_create_file_non_atomic, [directory] * 100)

    # Check that the file was created
    assert len(directory.listdir()) < 100


##################################################
# Directories
##################################################


def test_create_unique_directory(tmpdir):
    directory = tmpdir.mkdir("test")

    # Check that there are no files in the directory
    assert len(directory.listdir()) == 0

    # Create a directory
    new_dir = create_unique_directory(directory, "test")
    assert os.path.basename(new_dir) == "test"
    assert os.path.exists(new_dir)

    # Check that the directory was created
    assert len(directory.listdir()) == 1


def test_create_unique_directory_duplicate_name(tmpdir):
    directory = tmpdir.mkdir("test")

    # Create a directory
    new_dir1 = create_unique_directory(directory, "test")
    new_dir2 = create_unique_directory(directory, "test")
    new_dir3 = create_unique_directory(directory, "test")

    assert new_dir1 != new_dir2
    assert new_dir1 != new_dir3
    assert new_dir2 != new_dir3

    assert os.path.basename(new_dir1) == "test"
    assert os.path.basename(new_dir2) == "test-1"
    assert os.path.basename(new_dir3) == "test-2"

    # Check that the directory was created
    assert len(directory.listdir()) == 3


def test_create_unique_directory_race_condition(tmpdir):
    """ Run `create_unique_directory` many times in parallel to make sure different processes don't accidentally try to use the same directory name. """
    directory = tmpdir.mkdir("test")

    with Pool(10) as p:
        p.map(_create_directory, [directory] * 100)

    # Check that the directory was created
    assert len(directory.listdir()) == 100
