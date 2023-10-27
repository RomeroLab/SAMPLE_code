""" Collection of methods for reading in sequence, T50 data.

Users will usually call get_dataset() without regard to the other functions.

Available functions:
    read_data: Reads Phil's .data files.
    read_alignment: Loads block alignment files.
    chimeras_to_onehot: Converts chimera sequences to one-hot encoding.
    get_dataset: Parses the necessary files to read in a dataset.
"""

import tools


def read_data(datafile):
    """Reads the standard data file format I (Phil) have been using. Taken from
    'https://github.com/RomeroLab/useful-scripts/blob/master/phils_python_modul
    es/data_tools.py.

    Args:
        datafile: Name of datafile to be read from.

    Returns:
        header: Beginning information in datafile.
        data: List of tuples derived from comma-separated rows of datafile.
    """
    data = open(datafile, 'r').read().strip().replace('\t', '').split('\n')

    # remove all comments in data file
    nocomments = [line.split('#')[0].strip() for line in data if
                  len(line.strip()) > 0 and line.strip()[0] != '#']

    # get the header
    header_ind = [i for i, row in enumerate(nocomments)
                  if '>header' in row][0] + 1
    header = [h.strip() for h in nocomments[header_ind].split(',')]

    # get the data
    data_ind = [i for i, row in enumerate(nocomments) if '>data' in row][0] + 1
    data = []
    for line in nocomments[data_ind:]:
        d = [e.strip() for e in line.split(',')]
        data.append(d)
    return header, data


def read_alignment(filename):
    """Reads in alignment file.

    Args:
        filename: Name of fasta alignment file to be read.

    Returns:
        alignment: List of tuples corresponding to parental amino acids at each
            position.
        seq_names: Names of the parent sequences.
    """
    file = open(filename).read()
    data = [line for line in file.split('\n') if len(line) > 0
            and line[0] != '#']
    if '>seq_names' in file:
        seq_names = data[data.index('>seq_names')+1:data.index('>alignment')]
    else:
        seq_names = []
    alignment_data = data[data.index('>alignment')+1:]
    alignment = [pos.split()[1:] for pos in alignment_data]
    return alignment, seq_names


def chimeras_to_onehot(dataset, block_aln):
    """Converts chimera sequences to one-hot encoded sequences using a parental
    block alignment.

    Args:
        dataset: Chimeric sequences and T50s read in by load_data().
        block_aln: Block alignment read in by read_alignment().

    Returns:
        oh_dataset: List with one-hot encoded sequences and associated T50s.
        blk_pos_AA_pars: List of tuples denoting the block, pos, amino acids,
            and parent associated with each one-hot encoding sequence position.
    """
    blk_pos_AA_pars = []
    for pos, block_AAs in enumerate(block_aln):
        AAs = block_AAs[1:]
        blk = int(block_AAs[0])
        for AA in set(AAs):
            parents = [i+1 for i, x in enumerate(AAs) if x == AA]
            blk_pos_AA_pars.append((blk, pos, AA, parents))

    oh_dataset = []
    for index, (chimera_seq, T50) in enumerate(dataset):
        one_hot = []
        for blk, pos, AA, pars in blk_pos_AA_pars:
            if int(chimera_seq[blk]) in pars:
                one_hot.append(1)
            else:
                one_hot.append(0)
        oh_dataset.append(tools.DP(one_hot, T50, index))
    return oh_dataset, blk_pos_AA_pars


def get_dataset(data_name, sorted_pos=True):
    """Returns a dataset composed of positive sequences with T50 values from a
    thermostability datafile and negative sequences from a binary
    functionality datafile.

    Args:
        data_name: Name of dataset to be read. Currently "P450","CBHII", or
            "sample_CBHII".
        sorted_pos: Whether the positive dataset will be sorted by T50 values
            before being returned.

    Returns:
        dataset: List with one-hot encoded sequences and associated T50s.
        one_hot_encoding: List of tuples denoting the block, pos, amino acids,
            and parent associated with each one-hot encoding sequence position.
    """
    # load thermostability data, Note: thermostability data is a subset of
    # function data
    names, thermo_data = read_data(f'{data_name}_thermostability.data')

    if data_name.split('_')[0] == 'sample':
        data_name = ''.join(data_name.split('_')[1:])

    # load binary function data
    names, function_data = read_data(f'{data_name}_function.data')

    block_aln, aln_cols = read_alignment(f'{data_name}_block_alignment.aln')

    # combine data sets to have a full active/inative and thermostability for
    # active data set
    positive = [(c[0], float(c[1])) for c in thermo_data]
    negative = [(c[0], float('NaN')) for c in function_data if c[1] == '0']
    for p in positive:
        for n in negative:
            try:
                assert p[0] != n[0]
            except AssertionError:
                print(p, n)
                raise
    if sorted_pos:
        sorted_positive = sorted(positive, reverse=True, key=lambda x: x[1])
        dataset = sorted_positive + negative
    else:
        dataset = positive + negative
    dataset, one_hot_encoding = chimeras_to_onehot(dataset, block_aln)

    return dataset, one_hot_encoding